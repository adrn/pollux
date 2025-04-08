"""Custom type hints for Pollux."""

from collections.abc import Callable
from typing import Any, Concatenate, TypeAlias, TypeVar

import jax
from jax.example_libraries.optimizers import Optimizer
from jaxtyping import Array, Float
from numpyro.optim import _NumPyroOptim
from typing_extensions import ParamSpec

LatentsT = Float[Array, "latents"]
DataT = Float[Array, "output"]

QuadT = Float[Array, "output latents latents"]
LinearT = Float[Array, "output latents"]
OutputT = Float[Array, "output"]

BatchedDataT = Float[Array, "#stars output"]
BatchedLatentsT = Float[Array, "#stars latents"]
AnyShapeFloatT = Float[Array, "..."]
BatchedOutputT = Float[Array, "#stars output"]

P = ParamSpec("P")
T = TypeVar("T")

# Define a type that can be either LatentsT only or LatentsT with other arguments
TransformFuncT: TypeAlias = (
    Callable[[LatentsT], OutputT] | Callable[Concatenate[LatentsT, P], OutputT]
)

OptimizerT = _NumPyroOptim | Optimizer | Any

PackedParamsT = dict[str, jax.Array]
UnpackedParamsT = dict[str, dict[str, jax.Array] | jax.Array]

__all__ = [
    "AnyShapeFloatT",
    "BatchedDataT",
    "BatchedLatentsT",
    "BatchedOutputT",
    "DataT",
    "LatentsT",
    "LinearT",
    "OptimizerT",
    "OutputT",
    "PackedParamsT",
    "QuadT",
    "TransformFuncT",
    "UnpackedParamsT",
]
