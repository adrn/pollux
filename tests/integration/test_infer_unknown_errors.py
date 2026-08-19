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

    flux_pp = plx.data.ShiftScalePreprocessor.from_data(data["flux"])
    all_data = plx.data.PolluxData(
        flux=plx.data.OutputData(data["flux"], preprocessor=flux_pp),
        label=plx.data.OutputData(
            data["label"],
            err=data["label_err"],
            preprocessor=plx.data.ShiftScalePreprocessor.from_data(data["label"]),
        ),
    ).preprocess()

    # The scatter is fitted on the preprocessed scale, so a prior we can reason about
    # in flux units goes through the same transform the errors would
    s_prior_scale = flux_pp.transform_err(jnp.full(n_flux, 0.1))
    err_trans = plx.models.FunctionTransform(
        output_size=n_flux,
        transform=lambda err, s: jnp.sqrt(err**2 + s**2),
        priors={"s": dist.HalfNormal(s_prior_scale)},
        shapes={},
        vmap=False,
    )

    model = plx.LVM(latent_size=n_latents)
    model.register_output(
        "flux", plx.models.LinearTransform(output_size=n_flux), err_trans
    )
    model.register_output("label", plx.models.LinearTransform(output_size=n_labels))

    opt_pars, res = model.optimize(
        all_data,
        num_steps=50_000,
        rng_key=jax.random.PRNGKey(0),
        optimizer=numpyro.optim.Adam(1e-3),
        progress=False,
    )
    res.losses.block_until_ready()

    # ...and read back out of it, to compare against the errors that generated the data
    s_flux_units = flux_pp.inverse_transform_err(opt_pars["flux"]["err"]["s"])
    assert np.isclose(np.mean(s_flux_units), np.mean(data["flux_err"]), atol=0.05)
