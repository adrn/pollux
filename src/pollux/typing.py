"""Custom type hints for Pollux."""

from collections.abc import Callable
from typing import Any

from jax.example_libraries.optimizers import Optimizer
from jaxtyping import Array, Float
from numpyro.optim import _NumPyroOptim

LatentsT = Float[Array, "latents"]

QuadT = Float[Array, "output latents latents"]
LinearT = Float[Array, "output latents"]
OutputT = Float[Array, "output"]

BatchedDataT = Float[Array, "#stars output"]
BatchedLatentsT = Float[Array, "#stars latents"]
BatchedOutputT = Float[Array, "#stars output"]

TransformFuncT = Callable[..., OutputT]

OptimizerT = _NumPyroOptim | Optimizer | Any

PackedParamsT = dict[str, Any]
UnpackedParamsT = dict[str, dict[str, Any] | Array]

__all__ = [
    "BatchedDataT",
    "BatchedLatentsT",
    "BatchedOutputT",
    "LatentsT",
    "LinearT",
    "OptimizerT",
    "OutputT",
    "PackedParamsT",
    "QuadT",
    "TransformFuncT",
    "UnpackedParamsT",
]
