"""Custom type hints for Pollux."""

from collections.abc import Callable
from typing import Any

from jax.example_libraries.optimizers import Optimizer
from jaxtyping import Array, Float
from numpyro.optim import _NumPyroOptim

LatentsT = Float[Array, "latents"]
DataT = Float[Array, "output"]

QuadT = Float[Array, "output latents latents"]
LinearT = Float[Array, "output latents"]
OutputT = Float[Array, "output"]

BatchedDataT = Float[Array, "#stars output"]
BatchedLatentsT = Float[Array, "#stars latents"]
AnyShapeFloatT = Float[Array, "..."]
BatchedOutputT = Float[Array, "#stars output"]

TransformFuncT = Callable[..., OutputT]

OptimizerT = _NumPyroOptim | Optimizer | Any

PackedParamsT = dict[str, Any]
UnpackedParamsT = dict[str, dict[str, Any] | Any | tuple[Any, ...]]

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
