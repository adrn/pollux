import jax.numpy as jnp
import numpy as np

from pollux.models.transforms import (
    LinearTransform,
    OffsetTransform,
    TransformSequence,
)


def test_sequence():
    n_stars = 128
    n_latents = 32
    n_out = 8
    rng = np.random.default_rng(0)

    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=8),
            OffsetTransform(output_size=n_stars, vmap=False),
        )
    )

    latents = rng.random((n_stars, n_latents))

    A = rng.random((n_out, n_latents))
    b = rng.random((n_stars, n_out))

    tmp = np.array([A @ latents[i] for i in range(n_stars)])
    tmp += b

    tmp2 = trans.apply(jnp.array(latents), {"A": A}, {"b": b})

    assert np.allclose(tmp2, tmp)
