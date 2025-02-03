"""Custom type hints for Pollux."""

from collections.abc import Callable
from typing import Concatenate, TypeAlias

from jaxtyping import Array, Float
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
TransformFuncT: TypeAlias = Callable[Concatenate[LatentsT, P], OutputT]
