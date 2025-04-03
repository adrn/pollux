import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from integration_test_helpers import make_simulated_linear_data
from numpyro import distributions as dist

import pollux as plx


def test_infer_distance():
    """
    Simulate data in which each object / star has an unknown offset (in distance
    modulus) in one label (like an apparent magnitude, where stars live on a manifold in
    absolute magnitude).
    """

    n_stars = 2048
    n_labels = 5
    n_latents = 32
    rng = np.random.default_rng(123)
    data, truth = make_simulated_linear_data(
        n_stars=n_stars,
        n_latents=n_latents,
        n_labels=n_labels,
        n_flux=128,
        rng=rng,
    )

    # True distance moduli:
    true_dm = rng.normal(11.0, 3.0, size=(n_stars, 1))
    all_data = plx.data.PolluxData(
        flux=plx.data.OutputData(
            data["flux"],
            err=data["flux_err"],
            preprocessor=plx.data.ShiftScalePreprocessor.from_data(data["flux"]),
        ),
        label=plx.data.OutputData(
            data["label"] + true_dm,
            err=data["label_err"],
            # preprocessor=plx.data.ShiftScalePreprocessor.from_data(data["label"]),
        ),
    )

    model = plx.LuxModel(latent_size=n_latents)
    model.register_output("flux", transform=plx.models.LinearTransform(output_size=128))

    trans = plx.models.TransformSequence(
        transforms=(
            plx.models.LinearTransform(
                output_size=n_labels, param_priors={"A": dist.Normal()}
            ),
            plx.models.OffsetTransform(
                output_size=n_stars,
                vmap=False,
                param_priors={"b": dist.Normal(11.0, 3.0)},
            ),
        )
    )
    model.register_output("label", transform=trans)

    params = {"flux": {"A": truth["B"]}, "label": {"t0_A": truth["A"], "t1_b": true_dm}}
    test_out = model.predict_outputs(truth["latents"], params)
    assert jnp.allclose(test_out["flux"], truth["flux"], atol=1e-5)
    assert jnp.allclose(test_out["label"], truth["label"] + true_dm, atol=1e-5)

    opt_params, res = model.optimize(
        all_data,
        num_steps=10000,
        rng_key=jax.random.PRNGKey(0),
        optimizer=numpyro.optim.Adam(1e-3),
    )
    res.losses.block_until_ready()

    d_dm = opt_params["label"]["t1_b"] - true_dm
    assert jnp.isclose(jnp.mean(d_dm), 0.0, atol=1e-2)
    assert jnp.isclose(jnp.std(d_dm), 1.0, atol=1e-2)
