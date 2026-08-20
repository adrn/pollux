"""Custom type hints for Pollux."""

from collections.abc import Callable
from typing import Any

from jaxtyping import Array, Float

LatentsT = Float[Array, "latents"]

LinearT = Float[Array, "output latents"]
OutputT = Float[Array, "output"]

BatchedDataT = Float[Array, "#stars output"]
BatchedLatentsT = Float[Array, "#stars latents"]
BatchedOutputT = Float[Array, "#stars output"]

TransformFuncT = Callable[..., OutputT]

# Any numpyro optimizer, or a raw jax optimizer triple. Narrowing this is not
# worth it: numpyro's own optimizer base class is private, and a union with Any
# collapses to Any regardless.
OptimizerT = Any

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
    "TransformFuncT",
    "UnpackedParamsT",
]
