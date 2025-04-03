__all__ = [
    "AbstractTransform",
    "AffineTransform",
    "LinearTransform",
    "OffsetTransform",
    "QuadraticTransform",
    "TransformSequence",
]

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
    """Represents a shape that may contain named dimensions"""

    dims: tuple[str | int, ...]

    def resolve(self, dim_sizes: dict[str, int | None]) -> tuple[int, ...]:
        """Convert named dimensions to concrete sizes."""
        return tuple(
            dim_sizes[d] if isinstance(d, str) and dim_sizes[d] is not None else d  # type: ignore[misc]
            for d in self.dims
        )


ParamPriorsT: TypeAlias = ImmutableMap[str, dist.Distribution]
ParamShapesT: TypeAlias = ImmutableMap[str, ShapeSpec]


class AbstractTransform(eqx.Module):
    """Base class defining the transform interface."""

    output_size: int
    param_priors: ParamPriorsT = eqx.field(converter=ImmutableMap)
    param_shapes: ParamShapesT

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the transform to input latent vectors."""
        raise NotImplementedError

    def get_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Get expanded parameter priors."""
        priors = {}
        for name, prior in self.param_priors.items():
            shape = self.param_shapes[name].resolve(
                {
                    "output_size": self.output_size,
                    "latent_size": latent_size,
                    "data_size": data_size,
                }
            )
            priors[name] = prior.expand(shape)
        return ImmutableMap(**priors)


class AbstractAtomicTransform(AbstractTransform):
    """Mixin class providing common functionality for atomic transforms."""

    transform: TransformFuncT[Any]
    _param_names: tuple[str, ...] = eqx.field(init=False, repr=False)
    _transform: TransformFuncT[Any] = eqx.field(init=False, repr=False)
    vmap: bool = True

    def __post_init__(self) -> None:
        sig = inspect.signature(self.transform)
        self._param_names = tuple(sig.parameters.keys())[1:]  # skip first (latents)

        # Validate parameters
        if not self._param_names:
            msg = "transform must accept parameters"
            raise ModelValidationError(msg)

        # Set up vmap'd transform
        self._transform = (
            jax.vmap(self.transform, in_axes=(0, *([None] * len(self._param_names))))
            if self.vmap
            else self.transform
        )

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        try:
            arg_params = tuple(params[p] for p in self._param_names)
        except KeyError as e:
            msg = f"Missing parameters: {self._param_names}"
            raise RuntimeError(msg) from e
        return self._transform(latents, *arg_params)


class TransformSequence(AbstractTransform):
    """A sequence of transforms applied in order."""

    transforms: tuple[AbstractTransform, ...]

    def __init__(self, transforms: tuple[AbstractTransform, ...]):
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
        return ImmutableMap(
            **{
                f"t{i}_{name}": prior
                for i, transform in enumerate(transforms)
                for name, prior in transform.param_priors.items()
            }
        )

    @staticmethod
    def _combine_shapes(transforms: tuple[AbstractTransform, ...]) -> ParamShapesT:
        return ImmutableMap(
            **{
                f"t{i}_{name}": shape
                for i, transform in enumerate(transforms)
                for name, shape in transform.param_shapes.items()
            }
        )

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        output = latents
        for i, transform in enumerate(self.transforms):
            transform_params = {
                name.split("_")[1]: params[f"t{i}_{name.split('_')[1]}"]
                for name in params
                if name.startswith(f"t{i}_")
            }
            output = transform.apply(output, **transform_params)
        return output


# ----


def _linear_transform(z: LatentsT, A: LinearT) -> OutputT:
    return A @ z


class LinearTransform(AbstractAtomicTransform):
    transform: TransformFuncT[LinearT] = _linear_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {"A": ShapeSpec(("output_size", "latent_size"))}
    )


# ----


def _offset_transform(z: LatentsT, b: OutputT) -> OutputT:
    return z + b


class OffsetTransform(AbstractAtomicTransform):
    transform: TransformFuncT[LinearT] = _offset_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap({"b": ShapeSpec(("output_size",))})


# ----


def _affine_transform(z: LatentsT, A: LinearT, b: OutputT) -> OutputT:
    return A @ z + b


class AffineTransform(AbstractAtomicTransform):
    transform: TransformFuncT[LinearT, OutputT] = _affine_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"A": dist.Normal(0, 1), "b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap(
        {
            "A": ShapeSpec(("output_size", "latent_size")),
            "b": ShapeSpec(("output_size",)),
        }
    )


# ----


def _quadratic_transform(z: LatentsT, Q: QuadT, A: LinearT, b: OutputT) -> OutputT:
    return jnp.einsum("i,oij,j->o", z, Q, z) + A @ z + b


class QuadraticTransform(AbstractAtomicTransform):
    transform: TransformFuncT[QuadT, LinearT, OutputT] = _quadratic_transform
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
            "b": ShapeSpec(("output_size",)),
        }
    )


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
