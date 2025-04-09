"""
TODO: fix the docstrings and typing?
"""

__all__ = [
    "AbstractTransform",
    "AffineTransform",
    "FunctionTransform",
    "LinearTransform",
    "NoOpTransform",
    "OffsetTransform",
    "QuadraticTransform",
    "TransformSequence",
]

import abc
import inspect
from dataclasses import dataclass
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from xmmutablemap import ImmutableMap

from ..exceptions import ModelValidationError
from ..typing import (
    BatchedLatentsT,
    BatchedOutputT,
    LatentsT,
    LinearT,
    OutputT,
    QuadT,
    TransformFuncT,
)


@dataclass(frozen=True)
class ShapeSpec:
    """Represents a shape that may contain named dimensions."""

    dims: tuple[str | int, ...]

    def resolve(self, dim_sizes: dict[str, int | None]) -> tuple[int, ...]:
        """Convert named dimensions to concrete sizes.

        Uses the provided dimension size mappings to convert any string dimension
        names to their integer sizes.
        """
        dim_sizes = dim_sizes.copy()
        dim_sizes["one"] = 1
        return tuple(
            dim_sizes[d] if isinstance(d, str) and dim_sizes[d] is not None else d  # type: ignore[misc]
            for d in self.dims
        )


ParamPriorsT: TypeAlias = ImmutableMap[str, dist.Distribution]
ParamShapesT: TypeAlias = ImmutableMap[str, ShapeSpec]


class AbstractTransform(eqx.Module):
    """Base class defining the transform interface.

    Transforms convert latent vectors to observable quantities through parameterized
    functions. They define the mapping between latent space and output spaces.
    """

    output_size: int
    param_priors: ParamPriorsT = eqx.field(converter=ImmutableMap)
    param_shapes: ParamShapesT = eqx.field(converter=ImmutableMap)

    @abc.abstractmethod
    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors.

        Takes a batch of latent vectors and transforms them using the provided
        parameters to produce output values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        raise NotImplementedError


class AbstractAtomicTransform(AbstractTransform):
    """Base class providing common functionality for atomic transforms.

    Atomic transforms apply a single operation to convert latent vectors to outputs.
    """

    transform: TransformFuncT
    _param_names: tuple[str, ...] = eqx.field(init=False, repr=False)
    _transform: TransformFuncT = eqx.field(init=False, repr=False)
    vmap: bool = True

    def __post_init__(self) -> None:
        """Initialize transform parameters after object creation.

        Extracts parameter names from the transform function signature and sets up
        vectorized application if requested.
        """
        sig = inspect.signature(self.transform)
        self._param_names = tuple(sig.parameters.keys())[1:]  # skip first (latents)

        # Set up vmap'd transform
        self._transform = (
            jax.vmap(self.transform, in_axes=(0, *([None] * len(self._param_names))))
            if self.vmap
            else self.transform
        )

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors.

        Extracts the required parameters from the kwargs and applies the transform
        function to the latents, handling vectorization automatically.
        """
        try:
            arg_params = tuple(params[p] for p in self._param_names)
        except KeyError as e:
            msg = f"Missing parameters: {self._param_names}"
            raise RuntimeError(msg) from e
        return self._transform(latents, *arg_params)

    def get_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        priors = {}
        for name, prior in self.param_priors.items():
            if name in self.param_shapes:
                shape = self.param_shapes[name].resolve(
                    {
                        "output_size": self.output_size,
                        "latent_size": latent_size,
                        "data_size": data_size,
                    }
                )
                priors[name] = prior.expand(shape)
            else:
                priors[name] = prior
        return ImmutableMap(**priors)


class TransformSequence(AbstractTransform):
    """A sequence of transforms applied in order.

    Composes multiple transforms together, where the output of each transform
    becomes the input to the next transform in the sequence.
    """

    transforms: tuple[AbstractTransform, ...]

    def __init__(self, transforms: tuple[AbstractTransform, ...]):
        """Initialize a sequence of transforms.

        Combines the parameter priors and shapes from all transforms in the sequence,
        prefixing them with indices to maintain uniqueness.
        """
        if not transforms:
            msg = "At least one transform required"
            raise ModelValidationError(msg)

        super().__init__(
            output_size=transforms[-1].output_size,
            param_priors=self._combine_priors(transforms),
            param_shapes=self._combine_shapes(transforms),
        )
        self.transforms = transforms

    @staticmethod
    def _combine_priors(transforms: tuple[AbstractTransform, ...]) -> ParamPriorsT:
        """Combine parameter priors from multiple transforms."""
        return ImmutableMap(
            **{
                f"t{i}_{name}": prior
                for i, transform in enumerate(transforms)
                for name, prior in transform.param_priors.items()
            }
        )

    @staticmethod
    def _combine_shapes(transforms: tuple[AbstractTransform, ...]) -> ParamShapesT:
        """Combine parameter shapes from multiple transforms."""
        return ImmutableMap(
            **{
                f"t{i}_{name}": shape
                for i, transform in enumerate(transforms)
                for name, shape in transform.param_shapes.items()
            }
        )

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the sequence of transforms to input latent vectors.

        Passes the input through each transform in sequence, routing the appropriate
        parameters to each transform based on the prefixed parameter names.
        """
        output = latents
        for i, transform in enumerate(self.transforms):
            transform_params = {
                name.split("_")[1]: params[f"t{i}_{name.split('_')[1]}"]
                for name in params
                if name.startswith(f"t{i}_")
            }
            output = transform.apply(output, **transform_params)
        return output

    def get_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        priors = {}
        for (name, prior), trans in zip(self.param_priors.items(), self.transforms):
            shape = self.param_shapes[name].resolve(
                {
                    "output_size": trans.output_size,
                    "latent_size": latent_size,
                    "data_size": data_size,
                }
            )
            priors[name] = prior.expand(shape)
        return ImmutableMap(**priors)


class FunctionTransform(AbstractAtomicTransform):
    """Function transformation using a user-defined function.

    This transform allows for arbitrary transformations defined by the user.

    Examples
    --------
    TODO: add in quadrature
    """


# ----


def _noop_transform(z: LatentsT) -> OutputT:
    """No-op transformation."""
    return z


class NoOpTransform(AbstractAtomicTransform):
    """No-op transformation."""

    output_size: int = 0
    transform: TransformFuncT = _noop_transform
    param_priors: ParamPriorsT = ImmutableMap()
    param_shapes: ParamShapesT = ImmutableMap()


# ----


def _linear_transform(z: LatentsT, A: LinearT) -> OutputT:
    """Apply a linear transformation.

    Computes the matrix product A @ z.
    """
    return A @ z


class LinearTransform(AbstractAtomicTransform):
    """Linear transformation from latent to output space.

    Implements the transformation: y = A @ z, where A is a matrix and z is a latent
    vector.
    """

    transform: TransformFuncT = _linear_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {"A": ShapeSpec(("output_size", "latent_size"))}
    )


# ----


def _offset_transform(z: LatentsT, b: OutputT) -> OutputT:
    """Apply an offset transformation.

    Adds a bias vector b to the input: z + b.
    """
    return z + b


class OffsetTransform(AbstractAtomicTransform):
    """Offset transformation that adds a bias vector to inputs.

    Implements the transformation: y = z + b, where b is a bias vector.
    """

    transform: TransformFuncT = _offset_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap({"b": ShapeSpec(("output_size", "one"))})


# ----


def _affine_transform(z: LatentsT, A: LinearT, b: OutputT) -> OutputT:
    """Apply an affine transformation.

    Computes a linear transformation followed by an offset: A @ z + b.
    """
    return A @ z + b


class AffineTransform(AbstractAtomicTransform):
    """Affine transformation combining linear transform and offset.

    Implements the transformation: y = A @ z + b, where A is a matrix,
    z is a latent vector, and b is a bias vector.
    """

    transform: TransformFuncT = _affine_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {
            "A": ShapeSpec(("output_size", "latent_size")),
            "b": ShapeSpec(("output_size", "one")),
        }
    )


# ----


def _quadratic_transform(z: LatentsT, Q: QuadT, A: LinearT, b: OutputT) -> OutputT:
    """Apply a quadratic transformation.

    Computes a quadratic form plus a linear term and an offset: z^T Q z + A @ z + b.
    """
    return jnp.einsum("i,oij,j->o", z, Q, z) + A @ z + b


class QuadraticTransform(AbstractAtomicTransform):
    """Quadratic transformation of latent vectors.

    Implements the transformation: y = z^T Q z + A @ z + b, where Q is a tensor,
    A is a matrix, z is a latent vector, and b is a bias vector.
    """

    transform: TransformFuncT = _quadratic_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap(
            {"Q": dist.Normal(0, 1), "A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}
        ),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {
            "Q": ShapeSpec(("output_size", "latent_size", "latent_size")),
            "A": ShapeSpec(("output_size", "latent_size")),
            "b": ShapeSpec(("output_size", "one")),
        }
    )


# ----

# TODO: implement a Gaussian Process transform using the tinygp library. The user should specify the kernel, and parameter priors for the kernel.
# class GaussianProcessTransform(AtomicTransformMixin, AbstractTransform):
#     transform: TransformFuncT = ...


# # TODO: implement a simple multi-layer perceptron transform with flax. The user should
# # construct a MLP instance so that the parameter priors work for all layer parameters,
# # and there is a way of executing the forward pass with the correct shapes. This might
# # need to be a class factory so that the user can specify the layer shapes and
# # activation functions and the class is constructed from that. Also, make it work with
# # numpyro.
# class MLPTransform(AtomicTransformMixin, AbstractTransform):
#     transform: TransformFuncT = eqx.field(
#         default=lambda z, A, b: A @ z + b, init=False, repr=False
#     )
#     param_priors: TransformParamsT = eqx.field(
#         default_factory=_get_default_affine_priors,
#     )
