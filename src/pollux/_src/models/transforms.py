__all__ = [
    "AbstractOutputTransform",
    "AffineTransform",
    "LinearTransform",
    "QuadraticTransform",
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

    def resolve(self, dim_sizes: dict[str, int]) -> tuple[int, ...]:
        """Convert named dimensions to concrete sizes."""
        return tuple(dim_sizes[d] if isinstance(d, str) else d for d in self.dims)


ParamPriorsT: TypeAlias = ImmutableMap[str, dist.Distribution]
ParamShapesT: TypeAlias = ImmutableMap[str, ShapeSpec]


class AbstractOutputTransform(eqx.Module):
    output_size: int
    transform: eqx.AbstractVar[TransformFuncT[Any]]
    param_priors: eqx.AbstractVar[ParamPriorsT]
    param_shapes: eqx.AbstractVar[ParamShapesT]

    # The vmap'd transform function
    _transform: TransformFuncT[Any] = eqx.field(init=False, repr=False)

    # The parameter names, but preserving the order of parameter names that appear in
    # the call signature of the transform function
    _param_names: tuple[str, ...] = eqx.field(init=False, repr=False)

    # Special parameter names that can be specified
    _special_param_names: tuple[str] = ("s",)

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
        self._transform = jax.vmap(
            self.transform, in_axes=(0, *tuple([None] * len(self._param_names)))
        )

    # TODO: could dispatch so that LatentsT uses self.transform and BatchedLatentsT uses
    # the vmap'd transform
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

    def get_priors(self, latent_size: int) -> ParamPriorsT:
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
                    {"output_size": self.output_size, "latent_size": latent_size}
                )
                priors[name] = prior.expand(shape)
            else:
                priors[name] = prior
        return priors


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
