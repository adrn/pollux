"""Tests for the Cannon architecture."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import pollux as plx
import pollux.exceptions as plx_exceptions
from pollux._linalg import weighted_least_squares
from pollux.models import Cannon
from pollux.models.iterative import _latents_from_data
from pollux.models.transforms import (
    LinearTransform,
    NoOpTransform,
    PolyFeatureTransform,
    ScatterTransform,
    TransformSequence,
)

jax.config.update("jax_enable_x64", True)


class TestCannonBasic:
    """Construction and the polynomial feature bookkeeping."""

    def test_init(self):
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
        assert cannon.latent_size == 3  # the latents are the labels
        assert cannon.poly_degree == 2
        assert cannon.include_bias is True
        assert isinstance(cannon, plx.LVM)

    @pytest.mark.parametrize(
        ("label_size", "poly_degree", "include_bias", "expected"),
        [
            (3, 2, True, 10),  # C(3+2, 2)
            (2, 2, True, 6),  # C(2+2, 2)
            (3, 1, True, 4),  # C(3+1, 1)
            (3, 2, False, 9),  # C(3+2, 2) - 1
        ],
    )
    def test_n_features(self, label_size, poly_degree, include_bias, expected):
        cannon = Cannon(
            label_size=label_size,
            output_size=100,
            poly_degree=poly_degree,
            include_bias=include_bias,
        )
        assert cannon.n_features == expected

    def test_get_features(self):
        cannon = Cannon(label_size=2, output_size=10, poly_degree=2)
        labels = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        features = cannon.get_features(labels)

        assert features.shape == (2, 6)
        # For degree 2 with bias: [1, x1, x2, x1^2, x1*x2, x2^2]
        assert jnp.allclose(features[0], jnp.array([1.0, 1.0, 2.0, 1.0, 2.0, 4.0]))


class TestCannonStructure:
    """The registered outputs are what make this the Cannon."""

    def test_registers_labels_and_output(self):
        cannon = Cannon(label_size=3, output_size=100)
        assert sorted(cannon.outputs) == ["flux", "label"]

    def test_labels_are_the_latents(self):
        """A NoOpTransform on the labels is what ties the latents to them."""
        cannon = Cannon(label_size=3, output_size=100)
        assert isinstance(cannon.outputs["label"].data_transform, NoOpTransform)

    def test_output_is_poly_then_linear(self):
        cannon = Cannon(label_size=3, output_size=100, poly_degree=3)
        transform = cannon.outputs["flux"].data_transform
        assert isinstance(transform, TransformSequence)
        poly, linear = transform.transforms
        assert isinstance(poly, PolyFeatureTransform)
        assert poly.degree == 3
        assert isinstance(linear, LinearTransform)
        assert linear.output_size == 100

    def test_custom_output_names(self):
        cannon = Cannon(
            label_size=2, output_size=8, label_name="params", output_name="spec"
        )
        assert sorted(cannon.outputs) == ["params", "spec"]

    def test_same_name_twice_raises(self):
        """register_output already refuses a duplicate; no extra guard needed."""
        with pytest.raises(ValueError, match="already exists"):
            Cannon(label_size=2, output_size=8, label_name="x", output_name="x")

    def test_scatter_on_the_spectrum_by_default(self):
        """The per-pixel s_lambda is fitted; the catalog label errors are not."""
        cannon = Cannon(label_size=3, output_size=100)
        assert isinstance(cannon.outputs["flux"].err_transform, ScatterTransform)
        assert isinstance(cannon.outputs["label"].err_transform, NoOpTransform)

    def test_scatter_on_both_when_asked(self):
        cannon = Cannon(label_size=3, output_size=100, intrinsic_scatter=True)
        for output in cannon.outputs.values():
            assert isinstance(output.err_transform, ScatterTransform)

    def test_scatter_can_be_selected(self):
        cannon = Cannon(label_size=3, output_size=100, intrinsic_scatter=["flux"])
        assert isinstance(cannon.outputs["flux"].err_transform, ScatterTransform)
        assert isinstance(cannon.outputs["label"].err_transform, NoOpTransform)

    def test_scatter_prior_scale_from_mapping(self):
        cannon = Cannon(
            label_size=3, output_size=100, intrinsic_scatter={"label": 0.1, "flux": 1.0}
        )
        assert np.isclose(cannon.outputs["label"].err_transform.priors["s"].scale, 0.1)
        assert np.isclose(cannon.outputs["flux"].err_transform.priors["s"].scale, 1.0)

    def test_coeff_prior_is_the_regularization_knob(self):
        cannon = Cannon(label_size=3, output_size=8, coeff_prior=dist.Normal(0.0, 0.5))
        prior = cannon.outputs["flux"].data_transform.transforms[1].priors["A"]
        assert np.isclose(prior.scale, 0.5)


@pytest.fixture
def cannon_and_data():
    """A Cannon plus data generated from known coefficients.

    No intrinsic scatter, so the weights in the closed-form solve are exactly the
    inverse variances of the reported errors.
    """
    rng = np.random.default_rng(42)
    n_stars, n_labels, n_flux = 200, 3, 25

    cannon = Cannon(
        label_size=n_labels, output_size=n_flux, poly_degree=2, intrinsic_scatter=False
    )
    labels = jnp.array(rng.normal(size=(n_stars, n_labels)))
    features = cannon.get_features(labels)
    theta = jnp.array(rng.normal(size=(n_flux, cannon.n_features)))
    flux_err = jnp.array(rng.uniform(0.02, 0.2, size=(n_stars, n_flux)))
    flux = features @ theta.T + jnp.array(rng.normal(size=flux_err.shape)) * flux_err

    data = plx.data.PolluxData(
        label=plx.data.OutputData(labels, err=jnp.full((n_stars, n_labels), 1e-3)),
        flux=plx.data.OutputData(flux, err=flux_err),
    )
    return {
        "cannon": cannon,
        "data": data,
        "labels": labels,
        "features": features,
        "theta": theta,
        "flux": flux,
        "flux_err": flux_err,
    }


def _train(cannon, data, labels):
    """The classic Cannon training step: pin the latents, solve the coefficients."""
    return cannon.optimize_iterative(
        data,
        blocks=["flux:data"],
        fixed_pars={"latents": labels},
        max_cycles=1,
        progress=False,
    )


# The polynomial feature expansion makes the output non-affine in the latents, so any
# block that solves for the latents is downgraded from a closed form to gradient
# descent -- correctly, and loudly. Cannon users will meet this warning too.
expect_latents_fallback = pytest.mark.filterwarnings(
    "ignore:optimize_iterative could not use closed-form:"
    "pollux.exceptions.PolluxLinearizationWarning"
)


class TestCannonTraining:
    def test_matches_per_pixel_weighted_least_squares(self, cannon_and_data):
        """The framework must land on the same closed form the old fit() solved.

        This is the whole justification for rebuilding the Cannon on LVM: the
        per-pixel solve is still an exact weighted least squares, now reached through
        optimize_iterative's linear-block detection.
        """
        c = cannon_and_data
        cannon, features = c["cannon"], c["features"]

        # LinearTransform's default Normal(0, 1) prior is a ridge term of strength 1.0,
        # which is what optimize_iterative derives from it.
        reg = 1.0 * jnp.eye(cannon.n_features)
        ivar = 1.0 / c["flux_err"] ** 2
        expected = jax.vmap(lambda y, w: weighted_least_squares(features, y, w, reg))(
            c["flux"].T, ivar.T
        )

        res = _train(cannon, c["data"], c["labels"])
        assert jnp.allclose(res.params["flux"]["data"][1]["A"], expected)

    def test_recovers_known_coefficients(self, cannon_and_data):
        c = cannon_and_data
        res = _train(c["cannon"], c["data"], c["labels"])
        fitted = res.params["flux"]["data"][1]["A"]
        assert fitted.shape == (25, c["cannon"].n_features)
        assert jnp.allclose(fitted, c["theta"], atol=0.1)

    def test_a_narrower_coeff_prior_shrinks_the_coefficients(self, cannon_and_data):
        """Regularization is the prior on the coefficients."""
        c = cannon_and_data
        loose = _train(c["cannon"], c["data"], c["labels"])
        tight_cannon = Cannon(
            label_size=3,
            output_size=25,
            poly_degree=2,
            intrinsic_scatter=False,
            coeff_prior=dist.Normal(0.0, 1e-2),
        )
        tight = _train(tight_cannon, c["data"], c["labels"])

        loose_norm = jnp.sum(loose.params["flux"]["data"][1]["A"] ** 2)
        tight_norm = jnp.sum(tight.params["flux"]["data"][1]["A"] ** 2)
        assert tight_norm < loose_norm

    def test_predict_round_trip(self, cannon_and_data):
        c = cannon_and_data
        res = _train(c["cannon"], c["data"], c["labels"])
        pred = c["cannon"].predict_outputs(res.params)

        # The labels come back through the NoOpTransform untouched
        assert jnp.allclose(pred["label"], c["labels"])
        resid = pred["flux"] - c["flux"]
        assert jnp.sqrt(jnp.mean(resid**2)) < 0.2


class TestCannonLabelInference:
    """The other half of the Cannon: labels for stars with only a spectrum."""

    @expect_latents_fallback
    def test_infers_labels_with_output_params_fixed(self, cannon_and_data):
        c = cannon_and_data
        cannon, labels = c["cannon"], c["labels"]
        trained = _train(cannon, c["data"], labels)

        flux_only = plx.data.PolluxData(flux=c["data"]["flux"])
        res = cannon.optimize_iterative(
            flux_only,
            blocks=["latents"],
            fixed_pars=cannon.output_pars(trained.params),
            max_cycles=20,
            tol=1e-8,
            rng_key=jax.random.PRNGKey(0),
            progress=False,
        )

        inferred = res.params["latents"]
        assert inferred.shape == labels.shape

        # Labels are recovered from the spectra alone -- for nearly every star. A
        # degree-2 polynomial is not convex in the labels, so a minority of stars
        # settle into a different basin however long the optimizer runs, and an RMS
        # over all of them is dominated by those. Assert the bulk recovery instead,
        # which is what the model actually promises.
        err = jnp.abs(inferred - labels)
        assert jnp.median(err) < 0.05
        off = jnp.mean(jnp.max(err, axis=1) > 0.2)
        assert off < 0.1, f"{off:.1%} of stars found a different basin"

    def test_latents_block_is_not_closed_form(self, cannon_and_data):
        """The polynomial in the labels is exactly why: it is not affine in them."""
        c = cannon_and_data
        cannon = c["cannon"]
        flux_only = plx.data.PolluxData(flux=c["data"]["flux"])
        trained = _train(cannon, c["data"], c["labels"])

        with pytest.warns(
            plx_exceptions.PolluxLinearizationWarning, match="not affine in the latents"
        ):
            cannon.optimize_iterative(
                flux_only,
                blocks=["latents"],
                fixed_pars=cannon.output_pars(trained.params),
                max_cycles=1,
                rng_key=jax.random.PRNGKey(0),
                progress=False,
            )

    def test_warm_start_uses_the_observed_labels(self, cannon_and_data):
        """The NoOpTransform passthrough lets the latents start at the labels."""
        c = cannon_and_data
        observed = _latents_from_data(c["cannon"], c["data"])
        assert observed is not None
        assert jnp.allclose(observed, c["labels"])


class TestCannonWithScatter:
    """With the scatter on, it is fitted alongside everything else."""

    @expect_latents_fallback
    def test_default_blocks_include_the_scatters(self, cannon_and_data):
        c = cannon_and_data
        cannon = Cannon(
            label_size=3, output_size=25, poly_degree=2, intrinsic_scatter=True
        )
        res = cannon.optimize_iterative(
            c["data"], max_cycles=1, rng_key=jax.random.PRNGKey(0), progress=False
        )
        names = [b.name for b in res.blocks]
        assert "flux:err" in names
        assert res.params["flux"]["err"]["s"].shape == (25,)
        assert jnp.all(res.params["flux"]["err"]["s"] >= 0)

    @expect_latents_fallback
    def test_coefficient_block_still_gets_a_closed_form(self, cannon_and_data):
        """The polynomial-then-linear output stays exactly solvable."""
        c = cannon_and_data
        cannon = Cannon(label_size=3, output_size=25, poly_degree=2)
        res = cannon.optimize_iterative(
            c["data"], max_cycles=1, rng_key=jax.random.PRNGKey(0), progress=False
        )
        solvers = {b.name: b.optimizer for b in res.blocks}
        assert solvers["flux:data"] == "least_squares"


class TestCannonHighDegree:
    def test_high_degree_polynomial(self):
        """Degree 4 with 2 labels: C(2+4, 4) = 15 features."""
        cannon = Cannon(label_size=2, output_size=6, poly_degree=4)
        assert cannon.n_features == 15
        assert cannon.get_features(jnp.zeros((3, 2))).shape == (3, 15)
