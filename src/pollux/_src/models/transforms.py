"""
TODO: fix the docstrings and typing?
"""

__all__ = [
    "AbstractSingleTransform",
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
ParamShapesT: TypeAlias = ImmutableMap[str, ShapeSpec | tuple[int, ...]]

# Tuples of parameters for TransformSequence
ParamPriorsTupleT: TypeAlias = tuple[ParamPriorsT, ...]
ParamShapesTupleT: TypeAlias = tuple[ParamShapesT, ...]


class AbstractTransform(eqx.Module):
    """Base class defining the transform interface.

    Transforms convert latent vectors to observable quantities through parameterized
    functions. They define the mapping between latent space and output spaces.
    """

    output_size: int

    # TODO: param_priors and param_shapes must be defined on any abstract subclass, but
    # there is no way to define an abstract class property
    # param_priors: ParamPriorsT | ParamPriorsTupleT
    # param_shapes: ParamShapesT | ParamShapesTupleT

    @abc.abstractmethod
    def apply(self, latents: BatchedLatentsT, **pars: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors.

        Takes a batch of latent vectors and transforms them using the provided
        parameters to produce output values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        raise NotImplementedError


class AbstractSingleTransform(AbstractTransform):
    """Base class providing common functionality for atomic transforms.

    "Single" transforms apply a single operation to convert latent vectors to outputs.
    """

    param_priors: ParamPriorsT = eqx.field(converter=ImmutableMap)
    param_shapes: ParamShapesT = eqx.field(converter=ImmutableMap)
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

    def apply(self, latents: BatchedLatentsT, **pars: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors.

        Extracts the required parameters from the kwargs and applies the transform
        function to the latents, handling vectorization automatically.
        """
        try:
            arg_pars = tuple(pars[p] for p in self._param_names)
        except KeyError as e:
            msg = f"Missing parameters: {self._param_names}"
            raise RuntimeError(msg) from e
        return self._transform(latents, *arg_pars)

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors.

        Expands the parameter prior distributions to the concrete shapes needed
        for the transform, based on latent size and optional data size.
        """
        priors = {}
        for name, prior in self.param_priors.items():
            if name in self.param_shapes:
                shapespec = self.param_shapes[name]
                shape = (
                    shapespec.resolve(
                        {
                            "output_size": self.output_size,
                            "latent_size": latent_size,
                            "data_size": data_size,
                        }
                    )
                    if isinstance(shapespec, ShapeSpec)
                    else shapespec
                )
                priors[name] = prior.expand(shape)
            else:
                priors[name] = prior
        return ImmutableMap(**priors)

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:  # TODO: fix Any types
        """For compatibility with TransformSequence."""
        for param_name in self._param_names:
            if param_name not in flat_pars and not ignore_missing:
                msg = f"Missing value in transform: {param_name}"
                raise ValueError(msg)
        return flat_pars

    def pack_pars(
        self, nested_pars: dict[str, Any], ignore_missing: bool = False
    ) -> dict[str, Any]:  # TODO: fix Any types
        """For compatibility with TransformSequence."""
        for param_name in self._param_names:
            if param_name not in nested_pars and not ignore_missing:
                msg = f"Missing value in transform: {param_name}"
                raise ValueError(msg)
        return nested_pars


class TransformSequence(AbstractTransform):
    """A sequence of transforms applied in order.

    Composes multiple transforms together, where the output of each transform
    becomes the input to the next transform in the sequence.

    Parameters are stored as tuples of dictionaries, one element per transform.
    """

    transforms: tuple[AbstractSingleTransform, ...]

    def __init__(self, transforms: tuple[AbstractSingleTransform, ...]):
        """Initialize a sequence of transforms."""
        if not transforms:
            msg = "At least one transform required"
            raise ModelValidationError(msg)

        # Set output size to the final transform's output size
        self.output_size = transforms[-1].output_size
        self.transforms = transforms

    @property
    def param_priors(self) -> ParamPriorsTupleT:
        """Collect parameter priors from all transforms in the sequence."""
        return tuple(
            getattr(transform, "param_priors", ImmutableMap())
            for transform in self.transforms
        )

    @property
    def param_shapes(self) -> ParamShapesTupleT:
        """Collect parameter shapes from all transforms in the sequence."""
        return tuple(
            getattr(transform, "param_shapes", ImmutableMap())
            for transform in self.transforms
        )

    def apply(
        self, latents: BatchedLatentsT, *args: dict[str, Any], **kwargs: Any
    ) -> BatchedOutputT:
        """Apply the sequence of transforms to input latent vectors.

        Parameters can be provided in two ways:
        1. As positional arguments: One dictionary per transform in sequence order
        2. As keyword arguments: Using "{transform_index}:{param}" naming scheme, so a
           parameter named "A" in transform 0 of the sequence would be "0:A".

        Parameters
        ----------
        latents
            Input latent vectors
        *args
            Parameter dictionaries, one per transform in the sequence
        **kwargs
            Flat parameters using the new naming scheme "{transform_index}:{param_name}"
        """
        # Check if using positional parameter dictionaries
        if args:
            if len(args) != len(self.transforms):
                msg = (
                    f"Expected {len(self.transforms)} parameter dictionaries, "
                    f"got {len(args)}"
                )
                raise ValueError(msg)

            if kwargs:
                msg = "Cannot mix positional parameter dicts with keyword parameters"
                raise ValueError(msg)

            output = latents
            for transform, transform_pars in zip(self.transforms, args):
                output = transform.apply(output, **transform_pars)
            return output

        # Handle flat format with "{index}:{param}" naming
        transform_pars_list: list[dict[str, Any]] = [{} for _ in self.transforms]

        for param_name, param_value in kwargs.items():
            if ":" in param_name:
                # New format: "0:A", "1:p1", etc.
                idx_str, actual_param_name = param_name.split(":", 1)
                transform_idx = int(idx_str)
                if not 0 <= transform_idx < len(self.transforms):
                    msg = f"Invalid transform index: {transform_idx}"
                    raise ValueError(msg)
                transform_pars_list[transform_idx][actual_param_name] = param_value

            else:
                # Handle any other parameter format as needed
                msg = f"Unsupported parameter name format: {param_name}"
                raise ValueError(msg)

        output = latents
        for transform, transform_pars in zip(self.transforms, transform_pars_list):
            output = transform.apply(output, **transform_pars)
        return output

    def unpack_pars(
        self, flat_pars: dict[str, Any], ignore_missing: bool = False
    ) -> tuple[dict[str, Any], ...]:
        """Convert flat parameter names to nested tuple structure.

        Takes parameters with names like "0:A", "1:p1" and converts them to
        a list of parameter dictionaries: [{"A": value}, {"p1": value}]

        Parameters
        ----------
        flat_pars
            Dictionary with parameter names in format "{transform_index}:{param_name}"

        Returns
        -------
        list
            List of parameter dictionaries, one per transform in the sequence
        """
        nested_pars: list[dict[str, Any]] = [{} for _ in self.transforms]

        for param_name in self.names_flat:
            param_value = flat_pars.get(param_name)

            if param_value is None:
                if not ignore_missing:
                    msg = f"Missing value in transform: {param_name}"
                    raise ValueError(msg)
                # Skip missing parameters when ignore_missing=True
                continue

            if ":" in param_name:
                idx_str, actual_param_name = param_name.split(":", 1)
                transform_idx = int(idx_str)
                if 0 <= transform_idx < len(self.transforms):
                    nested_pars[transform_idx][actual_param_name] = param_value

        return tuple(nested_pars)

    def pack_pars(
        self, nested_pars: list[dict[str, Any]], ignore_missing: bool = False
    ) -> dict[str, Any]:
        """Convert nested parameter structure to flat naming scheme.

        Takes a list of parameter dictionaries and converts them to flat
        parameter names like "0:A", "1:p1".

        Parameters
        ----------
        nested_pars
            List of parameter dictionaries, one per transform in the sequence
        ignore_missing
            If True, skip missing parameters instead of raising an error.
            Currently unused but kept for API consistency.

        Returns
        -------
        dict
            Dictionary with parameter names in format "{transform_index}:{param_name}"
        """
        # TODO: ignore_missing is not used here...?
        _ = ignore_missing

        flat_pars = {}

        for i, transform_pars in enumerate(nested_pars):
            for param_name, param_value in transform_pars.items():
                flat_name = f"{i}:{param_name}"
                flat_pars[flat_name] = param_value

        return flat_pars

    @property
    def names_nested(self) -> tuple[tuple[str, ...], ...]:
        return tuple(t._param_names for t in self.transforms)

    @property
    def names_flat(self) -> tuple[str, ...]:
        return tuple(
            f"{i}:{name}" for i, names in enumerate(self.names_nested) for name in names
        )

    def get_expanded_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors using flat naming scheme.

        Returns flattened parameter priors with index-based naming for
        compatibility with the AbstractTransform interface.
        Parameter names will be in the format: "{transform_index}:{param_name}"
        """

        priors = {}
        for i, transform in enumerate(self.transforms):
            transform_priors = transform.get_expanded_priors(
                latent_size=latent_size, data_size=data_size
            )
            for param_name, prior in transform_priors.items():
                flat_name = f"{i}:{param_name}"
                priors[flat_name] = prior
        return ImmutableMap(**priors)


class FunctionTransform(AbstractSingleTransform):
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


class NoOpTransform(AbstractSingleTransform):
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


class LinearTransform(AbstractSingleTransform):
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


class OffsetTransform(AbstractSingleTransform):
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


class AffineTransform(AbstractSingleTransform):
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


class QuadraticTransform(AbstractSingleTransform):
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
# class GaussianProcessTransform(SingleTransformMixin, AbstractTransform):
#     transform: TransformFuncT = ...


# # TODO: implement a simple multi-layer perceptron transform with flax. The user should
# # construct a MLP instance so that the parameter priors work for all layer parameters,
# # and there is a way of executing the forward pass with the correct shapes. This might
# # need to be a class factory so that the user can specify the layer shapes and
# # activation functions and the class is constructed from that. Also, make it work with
# # numpyro.
# class MLPTransform(SingleTransformMixin, AbstractTransform):
#     transform: TransformFuncT = eqx.field(
#         default=lambda z, A, b: A @ z + b, init=False, repr=False
#     )
#     param_priors: TransformParamsT = eqx.field(
#         default_factory=_get_default_affine_priors,
#     )
