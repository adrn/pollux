import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import pollux as plx
from pollux.models.transforms import (
    LinearTransform,
    OffsetTransform,
    TransformSequence,
)


def test_transform_sequence():
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

    tmp2 = trans.apply(jnp.array(latents), t0_A=A, t1_b=b)

    assert np.allclose(tmp2, tmp)


def test_transform_sequence_priors():
    n_stars = 128
    n_latents = 32
    n_out = 8
    rng = np.random.default_rng(0)

    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=8, param_priors={"A": dist.Laplace()}),
            OffsetTransform(
                output_size=n_stars,
                vmap=False,
                param_priors={"b": dist.Normal(11.0, 3.0)},
            ),
        )
    )

    latents = jnp.array(rng.random((n_stars, n_latents)))

    A = rng.random((n_out, n_latents))
    b = rng.random((n_stars, n_out))

    tmp = np.array([A @ latents[i] for i in range(n_stars)])
    tmp += b

    params = {"mags": {"t0_A": A, "t1_b": b}}

    model = plx.LuxModel(latent_size=n_latents)
    model.register_output("mags", transform=trans)
    out = model.predict_outputs(latents, params)

    assert np.allclose(out["mags"], tmp)
