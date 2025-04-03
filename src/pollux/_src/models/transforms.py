__all__ = [
    "AbstractOutputTransform",
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
    """Base class for all transforms."""

    output_size: int

    # The vmap'd transform function
    _transform: TransformFuncT[Any] = eqx.field(init=False, repr=False)
    _param_names: tuple[str, ...] = eqx.field(init=False, repr=False)
    _special_param_names: tuple[str] = eqx.field(init=False, repr=False, default=("s",))

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the transform to the input latent vectors"""
        try:
            arg_params = (params[p] for p in self._param_names)
        except KeyError as e:
            msg = (
                "You must provide parameter values for all parameter names "
                f"({self._param_names})"
            )
            raise RuntimeError(msg) from e
        return self._transform(latents, *arg_params)

    def get_priors(
        self, latent_size: int, data_size: int | None = None
    ) -> ParamPriorsT:
        """Expand the numpyro priors to the expected shapes and return them.

        Parameters
        ----------
        latent_size
            The size of the latent vectors.
        """
        # TODO: this might need to be more general if we want to support, e.g., NN's
        # with hidden layers with their own sizes. Then need to maintain a list of
        # internal sizes or something, of which output_size is one.
        priors = {}
        for name, prior in self.param_priors.items():
            # Be more permissable with the shapes of special parameters:
            if name not in self._special_param_names:
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


class AbstractOutputTransform(AbstractTransform):
    """Base class for atomic transforms with fixed parameter structures."""

    transform: eqx.AbstractVar[TransformFuncT[Any]]
    param_priors: eqx.AbstractVar[ParamPriorsT]
    param_shapes: eqx.AbstractVar[ParamShapesT]
    vmap: bool = True

    def __post_init__(self) -> None:
        # Validate transform parameters match signature
        sig = inspect.signature(self.transform)
        all_args = tuple(sig.parameters.keys())
        self._param_names = all_args[1:]

        if len(all_args) < 1:
            msg = "transform must accept at least one argument (latent vector)"
            raise ModelValidationError(msg)

        # Skip first parameter (latent vector)
        required_params = set(self._param_names)
        allowed_params = required_params.union(set(self._special_param_names))
        provided_params = set(self.param_priors.keys())

        if required_params != provided_params:
            missing = required_params - provided_params
            extra = provided_params - allowed_params
            msgs = []
            if missing:
                msgs.append(f"Missing parameters: {missing}")
            if extra:
                msgs.append(f"Unexpected parameters: {extra}")

            if msgs:
                raise ModelValidationError(". ".join(msgs))

        # Validate all priors are proper distributions
        for name, prior in self.param_priors.items():
            if not isinstance(prior, dist.Distribution):
                msg = f"Prior '{name}' must be a numpyro Distribution instance"
                raise ModelValidationError(msg)

            if name not in self.param_shapes and name not in self._special_param_names:
                msg = f"Prior '{name}' must have a shape specification"
                raise ModelValidationError(msg)

        # now we make sure to vmap over stars:
        self._transform = (
            jax.vmap(
                self.transform, in_axes=(0, *tuple([None] * len(self._param_names)))
            )
            if self.vmap
            else self.transform
        )


class TransformSequence(AbstractTransform):
    """A sequence of output transforms."""

    transforms: tuple[AbstractOutputTransform, ...]

    _transform: TransformFuncT[Any] = eqx.field(init=False, repr=False)
    param_priors: ParamPriorsT = eqx.field(init=False)
    param_shapes: ParamShapesT = eqx.field(init=False)

    def __init__(
        self, transforms: tuple[AbstractOutputTransform, ...], vmap: bool = True
    ):
        if not transforms:
            msg = "At least one transform must be specified"
            raise ModelValidationError(msg)

        super().__init__(output_size=transforms[-1].output_size)
        self.transforms = transforms

        # Build parameter mappings
        all_priors = {}
        all_shapes = {}
        for i, transform in enumerate(self.transforms):
            for name, prior in transform.param_priors.items():
                prefixed_name = f"t{i}_{name}"
                all_priors[prefixed_name] = prior
            for name, shape in transform.param_shapes.items():
                prefixed_name = f"t{i}_{name}"
                all_shapes[prefixed_name] = shape

        self.param_priors = ImmutableMap(**all_priors)
        self.param_shapes = ImmutableMap(**all_shapes)

        def _transform(z: LatentsT, **params: Any) -> OutputT:
            output = z
            for i, transform in enumerate(self.transforms):
                transform_params = {
                    name.split("_")[1]: params[f"t{i}_{name.split('_')[1]}"]
                    for name in params
                    if name.startswith(f"t{i}_")
                }
                output = transform.apply(output, **transform_params)
            return output

        self._param_names = tuple(all_shapes.keys())
        self._transform = _transform

    def apply(self, latents: BatchedLatentsT, **params: Any) -> BatchedOutputT:
        """Apply the transform to the input latent vectors"""
        return self._transform(latents, **params)


# ----


def _linear_transform(z: LatentsT, A: LinearT) -> OutputT:
    return A @ z


class LinearTransform(AbstractOutputTransform):
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


class OffsetTransform(AbstractOutputTransform):
    transform: TransformFuncT[LinearT] = _offset_transform
    param_priors: ParamPriorsT = eqx.field(
        default=ImmutableMap({"b": dist.Normal(0, 1)}),
        converter=ImmutableMap,
    )
    param_shapes: ParamShapesT = ImmutableMap({"b": ShapeSpec(("output_size",))})


# ----


def _affine_transform(z: LatentsT, A: LinearT, b: OutputT) -> OutputT:
    return A @ z + b


class AffineTransform(AbstractOutputTransform):
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


class QuadraticTransform(AbstractOutputTransform):
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
# class GaussianProcessTransform(AbstractOutputTransform):
#     transform: TransformFuncT = ...


# # TODO: implement a simple multi-layer perceptron transform with flax. The user should
# # construct a MLP instance so that the parameter priors work for all layer parameters,
# # and there is a way of executing the forward pass with the correct shapes. This might
# # need to be a class factory so that the user can specify the layer shapes and
# # activation functions and the class is constructed from that. Also, make it work with
# # numpyro.
# class MLPTransform(AbstractOutputTransform):
#     transform: TransformFuncT = eqx.field(
#         default=lambda z, A, b: A @ z + b, init=False, repr=False
#     )
#     param_priors: TransformParamsT = eqx.field(
#         default_factory=_get_default_affine_priors,
#     )
