"""Custom type hints for Pollux."""

from typing import Protocol

import jax
import numpyro.distributions as dist

TransformParamsT = dict[str, dist.Distribution]


class TransformT(Protocol):
    def __call__(
        self, latent: jax.Array, *args: jax.Array, **kwargs: jax.Array
    ) -> jax.Array: ...
