import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from integration_test_helpers import make_simulated_linear_data

import pollux as plx


def test_infer_error_intrinsic_scatter():
    """
    Simulate data with uncertainty / intrinsic scatter, but pass in data with no error information and try to learn it.
    """

    n_stars = 2048
    n_labels = 2
    n_latents = 16
    n_flux = 128
    rng = np.random.default_rng(123)
    data, _truth = make_simulated_linear_data(
        n_stars=n_stars,
        n_latents=n_latents,
        n_labels=n_labels,
        n_flux=n_flux,
        rng=rng,
    )

    all_data = plx.data.PolluxData(
        flux=plx.data.OutputData(
            data["flux"],
            preprocessor=plx.data.ShiftScalePreprocessor.from_data(data["flux"]),
        ),
        label=plx.data.OutputData(
            data["label"],
            err=data["label_err"],
            preprocessor=plx.data.ShiftScalePreprocessor.from_data(data["label"]),
        ),
    )

    err_trans = plx.models.FunctionTransform(
        output_size=n_flux,
        transform=lambda err, s: jnp.sqrt(err**2 + s**2),
        param_priors={"s": dist.HalfNormal(0.1).expand((n_flux,))},
        param_shapes={},
        vmap=False,
    )

    model = plx.LuxModel(latent_size=n_latents)
    model.register_output(
        "flux", plx.models.LinearTransform(output_size=n_flux), err_trans
    )
    model.register_output("label", plx.models.LinearTransform(output_size=n_labels))

    opt_pars, res = model.optimize(
        all_data,
        num_steps=50_000,
        rng_key=jax.random.PRNGKey(0),
        optimizer=numpyro.optim.Adam(1e-3),
    )
    res.losses.block_until_ready()

    assert np.isclose(
        np.mean(opt_pars["flux"]["err"]["s"]), np.mean(data["flux_err"]), atol=0.05
    )
