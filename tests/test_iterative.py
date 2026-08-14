"""Tests for iterative optimization."""

import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux._linalg import weighted_least_squares
from pollux.exceptions import PolluxLinearizationWarning
from pollux.models.iterative import (
    IterativeOptimizationResult,
    ParameterBlock,
    _build_fixed_pars,
    _compute_loss,
    _get_regularization_from_prior,
    _inverse_variance,
    _latents_from_data,
    _latents_probe_points,
    _least_squares_blocker,
    _linearize_latents,
    _optimize_block_numpyro,
    _output_predict_fn,
    _participating_outputs,
    _solve_latents_least_squares,
    _solve_output_params_least_squares,
    _split_param_layer,
    optimize_iterative,
)
from pollux.models.transforms import (
    AffineTransform,
    FunctionTransform,
    LinearTransform,
    NoOpTransform,
    PolyFeatureTransform,
    ScatterTransform,
    TransformSequence,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def linear_model_and_data():
    """A LVM model with a single linear output, plus matching synthetic data."""
    n_stars = 32
    n_latents = 4
    n_flux = 16

    rng = np.random.default_rng(123)

    model = plx.LVM(latent_size=n_latents)
    model.register_output("flux", LinearTransform(output_size=n_flux))

    true_A = rng.normal(size=(n_flux, n_latents)) * 0.5
    true_latents = rng.normal(size=(n_stars, n_latents))
    true_flux = true_latents @ true_A.T
    flux_err = np.full_like(true_flux, 0.1)

    data = plx.data.PolluxData(
        flux=plx.data.OutputData(true_flux + rng.normal(0, flux_err), err=flux_err),
    )

    return {
        "model": model,
        "data": data,
        "true_A": true_A,
        "true_latents": true_latents,
        "n_stars": n_stars,
        "n_latents": n_latents,
        "n_flux": n_flux,
    }


class TestGetRegularizationFromPrior:
    """Tests for _get_regularization_from_prior helper."""

    def test_normal_prior_standard(self):
        """Normal(0, 1) should give regularization strength 1.0."""
        prior = dist.Normal(0.0, 1.0)
        reg_strength, prior_mean = _get_regularization_from_prior(prior)
        assert jnp.isclose(reg_strength, 1.0)
        assert jnp.isclose(prior_mean, 0.0)

    def test_normal_prior_custom_scale(self):
        """Normal(0, 0.5) should give regularization strength 4.0 (1/0.25)."""
        prior = dist.Normal(0.0, 0.5)
        reg_strength, prior_mean = _get_regularization_from_prior(prior)
        assert jnp.isclose(reg_strength, 4.0)
        assert jnp.isclose(prior_mean, 0.0)

    def test_normal_prior_nonzero_mean(self):
        """Normal(1.0, 2.0) should have mean 1.0 and regularization 0.25."""
        prior = dist.Normal(1.0, 2.0)
        reg_strength, prior_mean = _get_regularization_from_prior(prior)
        assert jnp.isclose(reg_strength, 0.25)
        assert jnp.isclose(prior_mean, 1.0)

    def test_improper_uniform_no_regularization(self):
        """ImproperUniform should give zero regularization."""
        prior = dist.ImproperUniform(dist.constraints.real, (), ())
        reg_strength, prior_mean = _get_regularization_from_prior(prior)
        assert jnp.isclose(reg_strength, 0.0)
        assert jnp.isclose(prior_mean, 0.0)


class TestLeastSquaresBlocker:
    """Which blocks get a closed-form solve is decided by structure, not by type."""

    def test_bare_linear_transform_is_solvable(self, linear_model_and_data):
        model, data = linear_model_and_data["model"], linear_model_and_data["data"]
        params = {"flux": {"data": {"A": jnp.array(linear_model_and_data["true_A"])}}}
        block = ParameterBlock("flux", "flux:data", optimizer="least_squares")
        assert _least_squares_blocker(model, data, params, block) is None

    def test_a_sequence_of_linear_pieces_is_solvable(self):
        """A composition used to be refused outright for being a TransformSequence."""
        n_stars, n_out = 8, 5
        model = plx.LVM(latent_size=4)
        model.register_output(
            "flux",
            TransformSequence(
                (_latent_slice(0, 2), LinearTransform(output_size=n_out))
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                jnp.ones((n_stars, n_out)), err=jnp.full((n_stars, n_out), 0.1)
            )
        )
        params = {
            "flux": {"data": ({}, {"A": jnp.zeros((n_out, 2))})},
            "latents": jnp.zeros((n_stars, 4)),
        }
        for spec in ("latents", "flux:data"):
            block = ParameterBlock(spec, spec, optimizer="least_squares")
            assert _least_squares_blocker(model, data, params, block) is None

    def test_error_transform_blocks_are_declined(self, linear_model_and_data):
        model, data = linear_model_and_data["model"], linear_model_and_data["data"]
        block = ParameterBlock("flux-err", "flux:err", optimizer="least_squares")
        reason = _least_squares_blocker(model, data, {}, block)
        assert reason is not None
        assert "error transform" in reason


class TestParameterBlock:
    """Tests for ParameterBlock dataclass."""

    def test_basic_creation(self):
        block = ParameterBlock(name="latents", params="latents")
        assert block.name == "latents"
        assert block.params == "latents"
        assert block.optimizer is None
        assert block.optimizer_kwargs == {}
        assert block.num_steps == 1000

    def test_with_least_squares(self):
        block = ParameterBlock(
            name="flux",
            params="flux:data",
            optimizer="least_squares",
        )
        assert block.optimizer == "least_squares"

    def test_with_optimizer_kwargs(self):
        block = ParameterBlock(
            name="labels",
            params="label:data",
            optimizer_kwargs={"step_size": 1e-3},
            num_steps=500,
        )
        assert block.optimizer_kwargs == {"step_size": 1e-3}
        assert block.num_steps == 500


def _latent_slice(lo, hi):
    """Route latents[lo:hi] to a branch. A closure, so the slice bounds stay out
    of the transform's signature -- FunctionTransform reads parameter names from it."""
    return FunctionTransform(output_size=hi - lo, transform=lambda z: z[lo:hi])


def _per_object_offset(output_size):
    """A fixed per-object additive offset, the distance-modulus recipe."""
    return FunctionTransform(
        output_size=output_size,
        transform=lambda y, offset: y + offset[:, None],
        priors={"offset": dist.Normal(0.0, 5.0)},
        shapes={"offset": ("data_size",)},
        vmap=False,
    )


class TestLinearizeLatents:
    """The linearization is exact where it claims to be, and declines where it isn't."""

    def test_recovers_the_design_matrix_bitwise(self, linear_model_and_data):
        """For a linear transform, the JVP *is* A -- not an approximation to it."""
        model = linear_model_and_data["model"]
        A = jnp.array(linear_model_and_data["true_A"])
        n_stars = linear_model_and_data["n_stars"]
        n_latents = linear_model_and_data["n_latents"]

        z0 = jnp.zeros((n_stars, n_latents))
        params = {"flux": {"data": {"A": A}, "err": {}}}
        c, jvp, _ = _linearize_latents(
            _output_predict_fn(model, "flux", params), z0, (jnp.ones_like(z0),)
        )

        columns = jnp.stack(
            [jvp(jnp.broadcast_to(e, z0.shape)) for e in jnp.eye(n_latents)], axis=-1
        )
        assert jnp.all(columns == A), "design matrix is not bitwise equal to A"
        assert jnp.all(c == 0), "offset is not bitwise zero"

    def test_offset_goes_into_c_and_not_the_design_matrix(self):
        """A per-object offset is constant in z, so it belongs entirely to c."""
        n_stars, n_latents, n_out = 8, 3, 5
        A = jax.random.normal(jax.random.PRNGKey(1), (n_out, n_latents))
        offset = jax.random.normal(jax.random.PRNGKey(2), (n_stars,))

        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            TransformSequence(
                (LinearTransform(output_size=n_out), _per_object_offset(n_out))
            ),
        )
        params = {"flux": {"data": ({"A": A}, {"offset": offset}), "err": {}}}

        z0 = jnp.zeros((n_stars, n_latents))
        c, jvp, _ = _linearize_latents(
            _output_predict_fn(model, "flux", params), z0, (jnp.ones_like(z0),)
        )

        assert jnp.allclose(c, offset[:, None])
        columns = jnp.stack(
            [jvp(jnp.broadcast_to(e, z0.shape)) for e in jnp.eye(n_latents)], axis=-1
        )
        assert jnp.allclose(columns, A)

    @pytest.mark.parametrize("scale", [0.1, 1.0, 100.0])
    def test_affine_verdict_holds_at_every_latent_scale(self, scale):
        """Linear stays linear and a 1e-4 nonlinearity stays detected, at any amplitude."""
        n_stars, n_latents, n_out = 8, 3, 5
        A = jax.random.normal(jax.random.PRNGKey(1), (n_out, n_latents))
        z0 = jnp.zeros((n_stars, n_latents))
        probes = _latents_probe_points(jnp.full(z0.shape, scale), z0.shape)

        # A tuple on success, a reason string on refusal -- so check the type, not
        # just "is not None", which a reason string would also satisfy
        assert not isinstance(_linearize_latents(lambda z: z @ A.T, z0, probes), str)
        assert isinstance(
            _linearize_latents(lambda z: z @ A.T + 1e-4 * (z**2) @ A.T, z0, probes),
            str,
        )

    def test_a_second_amplitude_catches_what_one_would_miss(self):
        """A nonlinearity built to vanish at one probe is still caught by the other."""
        n_stars, n_latents = 16, 3
        z0 = jnp.zeros((n_stars, n_latents))
        probes = _latents_probe_points(None, z0.shape)

        # sneaky(z) = z + z**2 (z - p) has tangent plane z at the origin, so its
        # deviation from it is z**3 - p z**2 -- engineered to vanish exactly at z = p
        # and to be large at 10p
        p = probes[0]

        def sneaky(z):
            return z + z**2 * (z - p)

        # A tuple on success, a reason string on refusal
        assert not isinstance(  # one probe is fooled
            _linearize_latents(sneaky, z0, probes[:1]), str
        )
        assert isinstance(_linearize_latents(sneaky, z0, probes), str)  # two are not

    def test_probe_points_track_the_current_latents(self):
        """Probes scale with the latents, never collapse to zero, and differ in size."""
        shape = (4, 2)
        for latents in (None, jnp.zeros(shape)):
            assert all(
                jnp.abs(p).max() > 0 for p in _latents_probe_points(latents, shape)
            )

        big = _latents_probe_points(jnp.full(shape, 50.0), shape)
        small = _latents_probe_points(jnp.full(shape, 0.5), shape)
        assert jnp.abs(big[0]).max() > 10 * jnp.abs(small[0]).max()
        # The amplitudes are spread, so a nearly-linear map is tested far from z0 too
        assert jnp.abs(small[1]).max() > 5 * jnp.abs(small[0]).max()


class TestSolveLatentsComposed:
    """The latents solve works on compositions, not just bare LinearTransforms."""

    def test_matches_the_explicit_normal_equations(self, linear_model_and_data):
        """Same answer as building A^T W A and A^T W y from A directly."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        A = jnp.array(linear_model_and_data["true_A"])
        n_latents = linear_model_and_data["n_latents"]

        params = {"flux": {"data": {"A": A}, "err": {}}}
        solved = _solve_latents_least_squares(model, data, params)

        w = 1.0 / data["flux"].err ** 2
        AtWA = jnp.einsum("nj,jk,jl->nkl", w, A, A) + jnp.eye(n_latents)
        AtWy = jnp.einsum("nj,jk,nj->nk", w, A, data["flux"].data)
        assert jnp.allclose(solved, jax.vmap(jnp.linalg.solve)(AtWA, AtWy))

    def test_partitioned_latents_recover_the_truth(self):
        """Half the latents feed one linear output, half feed another."""
        n_stars, n_out1, n_out2 = 256, 10, 3
        rng = np.random.default_rng(7)
        latents = jnp.array(rng.normal(size=(n_stars, 4)))
        A1 = jnp.array(rng.normal(size=(n_out1, 2)))
        A2 = jnp.array(rng.normal(size=(n_out2, 2)))

        model = plx.LVM(latent_size=4)
        for name, lo, hi, A in [("spec", 0, 2, A1), ("labels", 2, 4, A2)]:
            model.register_output(
                name,
                TransformSequence(
                    (_latent_slice(lo, hi), LinearTransform(output_size=A.shape[0]))
                ),
            )

        err1 = jnp.full((n_stars, n_out1), 1e-3)
        err2 = jnp.full((n_stars, n_out2), 1e-3)
        data = plx.data.PolluxData(
            spec=plx.data.OutputData(latents[:, :2] @ A1.T, err=err1),
            labels=plx.data.OutputData(latents[:, 2:] @ A2.T, err=err2),
        )
        params = {
            "spec": {"data": ({}, {"A": A1}), "err": {}},
            "labels": {"data": ({}, {"A": A2}), "err": {}},
        }

        solved = _solve_latents_least_squares(
            model, data, params, latents_prior=dist.Normal(0.0, 1e3)
        )
        assert jnp.allclose(solved, latents, atol=1e-2)

    def test_respects_a_fixed_per_object_offset(self):
        """Dropping c would bias every latent; the offset here is large enough to see."""
        n_stars, n_latents, n_out = 128, 3, 6
        rng = np.random.default_rng(11)
        latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
        A = jnp.array(rng.normal(size=(n_out, n_latents)))
        offset = jnp.array(rng.normal(size=n_stars) * 20.0)

        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            TransformSequence(
                (LinearTransform(output_size=n_out), _per_object_offset(n_out))
            ),
        )
        err = jnp.full((n_stars, n_out), 1e-3)
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(latents @ A.T + offset[:, None], err=err)
        )
        params = {"flux": {"data": ({"A": A}, {"offset": offset}), "err": {}}}

        solved = _solve_latents_least_squares(
            model, data, params, latents_prior=dist.Normal(0.0, 1e3)
        )
        assert jnp.allclose(solved, latents, atol=1e-2)

    def test_a_map_that_couples_objects_is_refused(self):
        """Affine is necessary but not sufficient.

        ``z - z.mean(axis=0)`` is perfectly affine and sails through the affineness
        probes, but its Jacobian is not block-diagonal, so there is no per-object
        solve to make. Accepting it returned confident nonsense: predictions off by
        ~1e13 rather than ~0.
        """
        n_stars, n_latents, n_out = 32, 3, 6
        rng = np.random.default_rng(0)
        latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
        A = jnp.array(rng.normal(size=(n_out, n_latents)))

        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            TransformSequence(
                (
                    FunctionTransform(
                        output_size=n_latents,
                        transform=lambda z: z - z.mean(axis=0),
                        vmap=False,
                    ),
                    LinearTransform(output_size=n_out),
                )
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                (latents - latents.mean(axis=0)) @ A.T,
                err=jnp.full((n_stars, n_out), 1e-3),
            )
        )
        params = {"latents": latents, "flux": {"data": ({}, {"A": A}), "err": {}}}

        reason = _solve_latents_least_squares(model, data, params)
        assert isinstance(reason, str)
        assert "couples objects" in reason

    def test_a_per_object_offset_is_not_mistaken_for_coupling(self):
        """The negative control: vmap=False is legitimate and must stay solvable."""
        n_stars, n_latents, n_out = 64, 3, 6
        rng = np.random.default_rng(2)
        latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
        A = jnp.array(rng.normal(size=(n_out, n_latents)))
        offset = jnp.array(rng.normal(size=n_stars))

        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            TransformSequence(
                (LinearTransform(output_size=n_out), _per_object_offset(n_out))
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                latents @ A.T + offset[:, None], err=jnp.full((n_stars, n_out), 1e-3)
            )
        )
        params = {"latents": latents, "flux": {"data": ({"A": A}, {"offset": offset})}}

        solved = _solve_latents_least_squares(
            model, data, params, latents_prior=dist.Normal(0.0, 1e3)
        )
        assert not isinstance(solved, str)
        assert jnp.allclose(solved, latents, atol=1e-2)

    def test_solvability_can_change_mid_fit_and_the_block_downgrades(self):
        """Affineness is a property of the parameters, not only of the model.

        f(z) = A z + q A z**2 is affine exactly when q == 0, so a block resolved as
        least-squares at q == 0 stops being solvable once an SVI block moves q. That
        used to raise ValueError partway through the fit.
        """
        n_stars, n_latents, n_out = 64, 3, 8
        rng = np.random.default_rng(0)
        latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
        A = jnp.array(rng.normal(size=(n_out, n_latents)))

        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            FunctionTransform(
                output_size=n_out,
                transform=jax.vmap(
                    lambda z, A, q: A @ z + q * (A @ z**2), in_axes=(0, None, None)
                ),
                priors={
                    "A": dist.Normal(0.0, 1.0).expand((n_out, n_latents)),
                    "q": dist.Normal(0.0, 1.0),
                },
                shapes={},
                vmap=False,
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(latents @ A.T, err=jnp.full((n_stars, n_out), 0.1))
        )
        # q starts at exactly zero, so the latents block resolves as least-squares
        initial = {
            "latents": latents,
            "flux": {"data": {"A": A, "q": jnp.array(0.0)}, "err": {}},
        }
        with pytest.warns(PolluxLinearizationWarning, match="no longer"):
            result = optimize_iterative(
                model,
                data,
                blocks=[
                    ParameterBlock("latents", "latents", optimizer="least_squares"),
                    ParameterBlock("flux:data", "flux:data", num_steps=200),
                ],
                initial_params=initial,
                max_cycles=3,
                rng_key=jax.random.PRNGKey(0),
                progress=False,
            )

        # The fit completed, and the block is permanently on SVI
        assert result.n_cycles >= 1
        assert {b.name: b.optimizer for b in result.blocks}["latents"] is None

    def test_nonlinear_output_is_refused_with_a_useful_message(self):
        """Polynomial features of the latents are not affine, so say so."""
        n_stars, n_latents, n_out = 16, 3, 5
        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            TransformSequence(
                (PolyFeatureTransform(degree=2), LinearTransform(output_size=n_out))
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                jnp.ones((n_stars, n_out)), err=jnp.full((n_stars, n_out), 0.1)
            )
        )
        A = jax.random.normal(jax.random.PRNGKey(0), (n_out, 10))
        params = {"flux": {"data": ({}, {"A": A}), "err": {}}}

        reason = _solve_latents_least_squares(model, data, params)
        assert isinstance(reason, str)
        assert "not affine in the latents" in reason


class TestSolveOutputParamsComposed:
    """The output-parameter solve uses the features reaching the linear layer."""

    def test_bare_linear_layer_is_unchanged(self, linear_model_and_data):
        """Same answer as solving directly against the latents."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        latents = jnp.array(linear_model_and_data["true_latents"])
        n_latents = linear_model_and_data["n_latents"]

        solved = _solve_output_params_least_squares(
            model, data, "flux", {"latents": latents}
        )

        w = 1.0 / data["flux"].err ** 2
        expected = jax.vmap(
            lambda y_dim, w_dim: weighted_least_squares(
                latents, y_dim, w_dim, jnp.eye(n_latents)
            )
        )(data["flux"].data.T, w.T)
        assert jnp.allclose(solved["A"], expected)

    def test_slice_prefix_recovers_the_coefficients(self):
        """Only the sliced latents should enter the design matrix."""
        n_stars, n_out = 256, 8
        rng = np.random.default_rng(3)
        latents = jnp.array(rng.normal(size=(n_stars, 5)))
        A = jnp.array(rng.normal(size=(n_out, 2)))

        model = plx.LVM(latent_size=5)
        model.register_output(
            "flux",
            TransformSequence(
                (_latent_slice(1, 3), LinearTransform(output_size=n_out))
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                latents[:, 1:3] @ A.T, err=jnp.full((n_stars, n_out), 1e-3)
            )
        )

        solved = _solve_output_params_least_squares(
            model, data, "flux", {"latents": latents}
        )
        # A tuple of per-child dicts, the layout a TransformSequence expects back
        assert isinstance(solved, tuple)
        assert solved[0] == {}
        assert jnp.allclose(solved[1]["A"], A, atol=1e-3)

    def test_polynomial_prefix_is_the_cannon(self):
        """Labels -> polynomial features -> linear, solved in closed form."""
        n_stars, n_out = 512, 6
        rng = np.random.default_rng(5)
        labels = jnp.array(rng.normal(size=(n_stars, 3)))
        n_features = 10  # 1 + 3 + 6 monomials up to degree 2
        coeffs = jnp.array(rng.normal(size=(n_out, n_features)))

        model = plx.LVM(latent_size=3)
        model.register_output(
            "flux",
            TransformSequence(
                (PolyFeatureTransform(degree=2), LinearTransform(output_size=n_out))
            ),
        )
        features = PolyFeatureTransform(degree=2).apply(labels)
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                features @ coeffs.T, err=jnp.full((n_stars, n_out), 1e-3)
            )
        )

        solved = _solve_output_params_least_squares(
            model, data, "flux", {"latents": labels}
        )
        assert solved[1]["A"].shape == (n_out, n_features)
        assert jnp.allclose(solved[1]["A"], coeffs, atol=1e-3)

    def test_affine_layer_solves_the_bias_jointly(self):
        """The bias becomes an extra column of ones in the design matrix."""
        n_stars, n_latents, n_out = 256, 3, 7
        rng = np.random.default_rng(13)
        latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
        A = jnp.array(rng.normal(size=(n_out, n_latents)))
        b = jnp.array(rng.normal(size=n_out))

        model = plx.LVM(latent_size=n_latents)
        model.register_output("flux", AffineTransform(output_size=n_out))
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                latents @ A.T + b, err=jnp.full((n_stars, n_out), 1e-3)
            )
        )

        solved = _solve_output_params_least_squares(
            model, data, "flux", {"latents": latents}
        )
        assert jnp.allclose(solved["A"], A, atol=1e-3)
        assert jnp.allclose(solved["b"], b, atol=1e-3)

    @pytest.mark.parametrize(
        "transform",
        [
            TransformSequence(
                (LinearTransform(output_size=4), PolyFeatureTransform(degree=2))
            ),
            TransformSequence(
                (LinearTransform(output_size=4), LinearTransform(output_size=4))
            ),
            PolyFeatureTransform(degree=2),
        ],
        ids=["linear-layer-not-last", "parameters-in-two-layers", "no-linear-layer"],
    )
    def test_unsolvable_shapes_are_declined(self, transform):
        assert _split_param_layer(transform) is None

    def test_unsolvable_output_raises_with_a_useful_message(self):
        model = plx.LVM(latent_size=3)
        model.register_output(
            "flux",
            TransformSequence(
                (LinearTransform(output_size=4), LinearTransform(output_size=4))
            ),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(jnp.ones((8, 4)), err=jnp.full((8, 4), 0.1))
        )
        with pytest.raises(ValueError, match="does not end in a linear layer"):
            _solve_output_params_least_squares(
                model, data, "flux", {"latents": jnp.zeros((8, 3))}
            )


class TestOptimizeIterativePartitionedLatents:
    """End to end: a model whose latent vector is split between two linear branches."""

    @pytest.fixture
    def partitioned_model_and_data(self):
        n_stars, n_spec, n_labels = 256, 12, 3
        rng = np.random.default_rng(17)
        latents = jnp.array(rng.normal(size=(n_stars, 4)))
        A_spec = jnp.array(rng.normal(size=(n_spec, 2)))
        A_labels = jnp.array(rng.normal(size=(n_labels, 2)))

        model = plx.LVM(latent_size=4)
        model.register_output(
            "spec",
            TransformSequence(
                (_latent_slice(0, 2), LinearTransform(output_size=n_spec))
            ),
        )
        model.register_output(
            "labels",
            TransformSequence(
                (_latent_slice(2, 4), LinearTransform(output_size=n_labels))
            ),
        )

        # Noiseless, so the test measures whether the solve finds the right subspace
        # rather than how much noise a 4-latent model can absorb
        data = plx.data.PolluxData(
            spec=plx.data.OutputData(
                latents[:, :2] @ A_spec.T, err=jnp.full((n_stars, n_spec), 1e-2)
            ),
            labels=plx.data.OutputData(
                latents[:, 2:] @ A_labels.T, err=jnp.full((n_stars, n_labels), 1e-2)
            ),
        )
        return model, data, latents, A_spec, A_labels

    def test_every_block_uses_the_closed_form_solve(self, partitioned_model_and_data):
        """This model ran entirely on Adam before: the branches are FunctionTransforms."""
        model, data, *_ = partitioned_model_and_data

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no fallback warning may be raised
            result = optimize_iterative(
                model,
                data,
                max_cycles=3,
                rng_key=jax.random.PRNGKey(0),
                progress=False,
            )

        assert [b.optimizer for b in result.blocks] == ["least_squares"] * 3
        assert [b.name for b in result.blocks] == [
            "latents",
            "spec:data",
            "labels:data",
        ]

    def test_recovers_the_generating_parameters(self, partitioned_model_and_data):
        model, data, latents, A_spec, A_labels = partitioned_model_and_data

        result = optimize_iterative(
            model,
            data,
            max_cycles=100,
            tol=1e-12,
            rng_key=jax.random.PRNGKey(0),
            latents_prior=dist.Normal(0.0, 1e3),
            progress=False,
        )

        # The factorization is only fixed up to a linear reparametrization within each
        # branch, so compare predictions rather than the individual factors.
        # Alternating least squares converges linearly on a bilinear problem, so the
        # tolerance is loose: what matters is that it heads for the right answer.
        predictions = model.predict_outputs(result.params)
        assert jnp.allclose(predictions["spec"], latents[:, :2] @ A_spec.T, atol=2e-2)
        assert jnp.allclose(
            predictions["labels"], latents[:, 2:] @ A_labels.T, atol=2e-2
        )
        assert result.losses_per_cycle[-1] < result.losses_per_cycle[0]


class TestErrTransformParticipates:
    """Error-transform parameters are part of the model, so they are part of the fit."""

    @pytest.fixture
    def scatter_model_and_data(self):
        n_stars, n_latents, n_flux = 64, 3, 12
        rng = np.random.default_rng(4)
        latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
        A = jnp.array(rng.normal(size=(n_flux, n_latents)))

        model = plx.LVM(latent_size=n_latents)
        model.register_output(
            "flux",
            LinearTransform(output_size=n_flux),
            err_transform=ScatterTransform(output_size=n_flux),
        )
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                latents @ A.T, err=jnp.full((n_stars, n_flux), 0.02)
            )
        )
        return model, data, latents, A

    def test_err_block_is_in_the_defaults(self, scatter_model_and_data):
        """It used to be left out, and then warned about by the very same function."""
        model, data, *_ = scatter_model_and_data

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = model.optimize_iterative(
                data, max_cycles=1, rng_key=jax.random.PRNGKey(0), progress=False
            )

        assert [b.name for b in result.blocks] == ["latents", "flux:data", "flux:err"]
        # The err block can never have a closed form, so it should not ask for one
        # and then be told no
        assert {b.name: b.optimizer for b in result.blocks}["flux:err"] is None
        assert not [w for w in record if "err_transform" in str(w.message)]
        assert result.params["flux"]["err"]["s"].shape == (data["flux"].data.shape[1],)

    def test_least_squares_weights_include_the_modelled_scatter(
        self, scatter_model_and_data
    ):
        """Weighting by raw errors while another block fits the scatter would leave
        the two blocks minimizing different objectives."""
        model, data, latents, _ = scatter_model_and_data
        n_flux = data["flux"].data.shape[1]

        # A scatter far larger than the reported errors, so ignoring it is obvious
        scatter = jnp.full(n_flux, 5.0)
        params = {"latents": latents, "flux": {"data": {}, "err": {"s": scatter}}}

        ivar = _inverse_variance(model, data, "flux", params)
        expected = 1.0 / (data["flux"].err ** 2 + scatter**2)
        assert jnp.allclose(ivar, expected)
        assert not jnp.allclose(ivar, 1.0 / data["flux"].err ** 2)

    def test_loss_keeps_the_normalization_when_the_variance_is_fitted(
        self, scatter_model_and_data
    ):
        """Without the log-determinant term the loss falls without bound as s grows."""
        model, data, latents, A = scatter_model_and_data
        base = {"latents": latents, "flux": {"data": {"A": A}}}

        small = _compute_loss(
            model,
            data,
            {**base, "flux": {**base["flux"], "err": {"s": jnp.full(12, 0.01)}}},
        )
        large = _compute_loss(
            model,
            data,
            {**base, "flux": {**base["flux"], "err": {"s": jnp.full(12, 100.0)}}},
        )
        assert large > small


class TestLatentsFromData:
    """Latents an output reports directly beat a draw from the prior."""

    @staticmethod
    def _data(**outputs):
        return plx.data.PolluxData(
            **{
                k: plx.data.OutputData(v, err=jnp.full(v.shape, 0.1))
                for k, v in outputs.items()
            }
        )

    def test_passthrough_output_supplies_the_latents(self):
        model = plx.LVM(latent_size=3)
        model.register_output("label", NoOpTransform())
        labels = jnp.arange(24.0).reshape(8, 3)
        data = self._data(label=labels)
        assert jnp.array_equal(_latents_from_data(model, data), labels)

    def test_a_transform_that_is_not_the_identity_is_not_used(self):
        """The gate is tested, not assumed: PolyFeatureTransform has no parameters
        either, but it is emphatically not a passthrough."""
        model = plx.LVM(latent_size=3)
        model.register_output("poly", PolyFeatureTransform(degree=2))
        data = self._data(poly=jnp.ones((8, 10)))
        assert _latents_from_data(model, data) is None

    def test_a_transform_with_parameters_is_not_used(self):
        """Its output only equals the latents for particular parameter values."""
        model = plx.LVM(latent_size=3)
        model.register_output("flux", LinearTransform(output_size=3))
        data = self._data(flux=jnp.ones((8, 3)))
        assert _latents_from_data(model, data) is None

    def test_a_passthrough_with_no_data_is_not_used(self):
        model = plx.LVM(latent_size=3)
        model.register_output("label", NoOpTransform())
        model.register_output("flux", LinearTransform(output_size=5))
        data = self._data(flux=jnp.ones((8, 5)))
        assert _latents_from_data(model, data) is None

    def test_default_blocks_fit_the_outputs_first_when_seeded(self):
        """Otherwise the first latents step chases prior-sampled output parameters
        and throws the head start away."""
        seeded = plx.LVM(latent_size=3)
        seeded.register_output("label", NoOpTransform())
        seeded.register_output("flux", LinearTransform(output_size=5))
        data = self._data(label=jnp.ones((8, 3)), flux=jnp.ones((8, 5)))
        result = seeded.optimize_iterative(
            data, max_cycles=1, rng_key=jax.random.PRNGKey(0), progress=False
        )
        assert [b.name for b in result.blocks] == ["flux:data", "latents"]

        # Without a passthrough output there is nothing to seed from, so the latents
        # keep going first
        plain = plx.LVM(latent_size=3)
        plain.register_output("flux", LinearTransform(output_size=5))
        result = plain.optimize_iterative(
            self._data(flux=jnp.ones((8, 5))),
            max_cycles=1,
            rng_key=jax.random.PRNGKey(0),
            progress=False,
        )
        assert [b.name for b in result.blocks] == ["latents", "flux:data"]


class TestCannonAsLux:
    """The Cannon written as a LVM model: labels are the latents, flux is poly->linear."""

    @pytest.fixture
    def cannon_model_and_data(self):
        n_stars, n_labels, n_flux = 200, 3, 40
        n_features = 10  # 1 + 3 + 6 monomials up to degree 2
        rng = np.random.default_rng(0)
        labels = jnp.array(rng.normal(size=(n_stars, n_labels)))
        theta = jnp.array(rng.normal(size=(n_flux, n_features)))
        flux = PolyFeatureTransform(degree=2).apply(labels) @ theta.T

        model = plx.LVM(latent_size=n_labels)
        model.register_output("label", NoOpTransform())
        model.register_output(
            "flux",
            TransformSequence(
                (PolyFeatureTransform(degree=2), LinearTransform(output_size=n_flux))
            ),
        )
        data = plx.data.PolluxData(
            label=plx.data.OutputData(labels, err=jnp.full((n_stars, n_labels), 1e-3)),
            flux=plx.data.OutputData(flux, err=jnp.full((n_stars, n_flux), 1e-2)),
        )
        return model, data, labels, theta

    def test_parameterless_output_gets_no_block(self, cannon_model_and_data):
        """The NoOpTransform label output has nothing to optimize, so skip it.

        It also has no entry in the parameter dict, which used to make the whole fit
        die with KeyError inside predict_outputs.
        """
        model, data, _, _ = cannon_model_and_data

        with pytest.warns(PolluxLinearizationWarning):
            result = model.optimize_iterative(
                data, max_cycles=2, rng_key=jax.random.PRNGKey(0), progress=False
            )

        # No block for "label", and the coefficients are fitted to the seeded
        # latents before the latents themselves are touched
        assert [b.name for b in result.blocks] == ["flux:data", "latents"]
        # Polynomial features of the latents are not affine in them, so the labels
        # cannot be solved in closed form -- but the coefficients still can be
        assert {b.name: b.optimizer for b in result.blocks} == {
            "latents": None,
            "flux:data": "least_squares",
        }

    def test_training_step_is_one_exact_solve(self, cannon_model_and_data):
        """With the labels known, fitting the Cannon is a single closed-form solve."""
        model, data, labels, theta = cannon_model_and_data

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no fallback, and no SVI
            result = model.optimize_iterative(
                data,
                blocks=["flux:data"],
                fixed_pars={"latents": labels},
                max_cycles=1,
                progress=False,
            )

        assert [b.optimizer for b in result.blocks] == ["least_squares"]
        assert jnp.allclose(result.params["flux"]["data"][1]["A"], theta, atol=1e-4)


class TestOptimizeIterative:
    """Tests for the main optimize_iterative function."""

    def test_optimize_iterative_basic(self, linear_model_and_data):
        """Test basic iterative optimization."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]

        result = optimize_iterative(
            model,
            data,
            max_cycles=3,
            rng_key=jax.random.PRNGKey(0),
        )

        assert isinstance(result, IterativeOptimizationResult)
        assert result.n_cycles == 3
        assert len(result.losses_per_cycle) == 3
        assert "latents" in result.params
        assert "flux" in result.params

    def test_optimize_iterative_with_custom_blocks(self, linear_model_and_data):
        """Test with custom parameter blocks."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]

        blocks = [
            ParameterBlock(
                name="latents",
                params="latents",
                optimizer="least_squares",
            ),
            ParameterBlock(
                name="flux",
                params="flux:data",
                optimizer="least_squares",
            ),
        ]

        result = optimize_iterative(
            model,
            data,
            blocks=blocks,
            max_cycles=5,
            rng_key=jax.random.PRNGKey(0),
        )

        # Should either converge or run all cycles
        assert result.n_cycles <= 5
        assert result.n_cycles >= 1
        # Loss should decrease over cycles for well-conditioned problem
        assert result.losses_per_cycle[-1] <= result.losses_per_cycle[0]

    def test_optimize_iterative_convergence(self, linear_model_and_data):
        """Test that optimization can converge."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]

        result = optimize_iterative(
            model,
            data,
            max_cycles=50,
            tol=1e-6,
            rng_key=jax.random.PRNGKey(0),
        )

        # Should converge before max_cycles for this simple problem
        # (or at least show decreasing loss)
        assert result.losses_per_cycle[-1] < result.losses_per_cycle[0]

    def test_optimize_iterative_with_initial_params(self, linear_model_and_data):
        """Test optimization with initial parameters."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        true_A = linear_model_and_data["true_A"]
        true_latents = linear_model_and_data["true_latents"]

        initial_params = {
            "latents": jnp.array(true_latents) + 0.1,  # Close to true
            "flux": {"data": {"A": jnp.array(true_A) + 0.1}, "err": {}},
        }

        result = optimize_iterative(
            model,
            data,
            initial_params=initial_params,
            max_cycles=5,
            rng_key=jax.random.PRNGKey(0),
        )

        # Should either converge or run all cycles
        assert result.n_cycles <= 5
        assert result.n_cycles >= 1
        # Providing initial params should work and produce finite losses
        assert all(jnp.isfinite(loss) for loss in result.losses_per_cycle)


class TestLuxOptimizeIterativeSignature:
    """The LVM.optimize_iterative method's parameter order."""

    def test_blocks_can_be_passed_positionally(self, linear_model_and_data):
        """`model.optimize_iterative(data, blocks)` binds blocks, not max_cycles."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]

        result = model.optimize_iterative(
            data,
            ["latents"],
            max_cycles=2,
            rng_key=jax.random.PRNGKey(0),
            progress=False,
        )
        assert isinstance(result, IterativeOptimizationResult)
        assert result.n_cycles <= 2
        assert "latents" in result.params

    def test_defaults_to_ten_cycles(self):
        """The method keeps its own max_cycles default, not the function's 100."""
        sig = inspect.signature(plx.LVM.optimize_iterative)
        assert sig.parameters["max_cycles"].default == 10
        assert list(sig.parameters)[:4] == ["self", "data", "blocks", "fixed_pars"]


class TestOptimizeBlockNumpyro:
    """Tests for the _optimize_block_numpyro function."""

    def test_svi_block_warm_starts_from_the_current_parameters(
        self, linear_model_and_data
    ):
        """A cycle must continue the previous one, not restart from the prior.

        AutoDelta re-initializes from its init strategy unless told otherwise, so an
        SVI block used to throw away every previous cycle's progress. That made the
        overall loss free to increase between cycles, which block coordinate descent
        must never do, and left models with a non-linear output stuck near their
        initialization no matter how many cycles were run.
        """
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        n_stars = linear_model_and_data["n_stars"]
        n_latents = linear_model_and_data["n_latents"]

        # A distinctive starting point, nowhere near any sensible initialization
        latents = jnp.full((n_stars, n_latents), 3.0)
        current = {
            "latents": latents,
            "flux": {"data": {"A": jnp.array(linear_model_and_data["true_A"])}},
        }

        block = ParameterBlock("latents", "latents", num_steps=1)
        stepped = _optimize_block_numpyro(
            model, data, block, current, jax.random.PRNGKey(0)
        )

        # One Adam step at the default 1e-3 can move each coordinate by ~1e-3
        assert jnp.abs(stepped["latents"] - latents).max() < 1e-2

    def test_build_fixed_pars_holds_everything_else(self, linear_model_and_data):
        """Parameters outside the optimized block are held fixed, not left free."""
        model = linear_model_and_data["model"]
        model.register_output("label", LinearTransform(output_size=2))

        current_params = {
            "latents": jnp.zeros((8, 4)),
            "flux": {"data": {"A": jnp.ones((16, 4))}, "err": {"b": jnp.ones((16, 1))}},
            "label": {"data": {"A": jnp.ones((2, 4))}, "err": {}},
        }
        fixed = _build_fixed_pars(model, current_params, ["flux:data"])

        # Latents and the other output are fixed entirely...
        assert jnp.allclose(fixed["latents"], current_params["latents"])
        assert set(fixed["label"]) == {"data", "err"}
        # ...as is the error transform of the output being optimized...
        assert jnp.allclose(fixed["flux"]["err"]["b"], jnp.ones((16, 1)))
        # ...but not the block itself
        assert "data" not in fixed["flux"]

    def test_optimize_block_numpyro_returns_params(self, linear_model_and_data):
        """Test that _optimize_block_numpyro returns valid parameters."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        true_A = linear_model_and_data["true_A"]
        true_latents = linear_model_and_data["true_latents"]

        # Initial params
        current_params = {
            "latents": jnp.array(true_latents),
            "flux": {"data": {"A": jnp.array(true_A) + 0.1}, "err": {}},
        }

        block = ParameterBlock(
            name="flux",
            params="flux:data",
            num_steps=100,
        )

        new_params = _optimize_block_numpyro(
            model,
            data,
            block,
            current_params,
            rng_key=jax.random.PRNGKey(0),
        )

        # Check that params were returned
        assert "flux" in new_params
        assert "data" in new_params["flux"]
        assert "A" in new_params["flux"]["data"]
        # Latents should be unchanged
        assert jnp.allclose(new_params["latents"], current_params["latents"])

    def test_optimize_block_numpyro_latents(self, linear_model_and_data):
        """Test that _optimize_block_numpyro can optimize latents."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        true_A = linear_model_and_data["true_A"]
        true_latents = linear_model_and_data["true_latents"]

        # Initial params with latents far from truth
        current_params = {
            "latents": jnp.zeros_like(true_latents),  # Start from zero
            "flux": {"data": {"A": jnp.array(true_A)}, "err": {}},
        }

        block = ParameterBlock(
            name="latents",
            params="latents",
            num_steps=200,
        )

        new_params = _optimize_block_numpyro(
            model,
            data,
            block,
            current_params,
            rng_key=jax.random.PRNGKey(0),
        )

        # Latents should have changed
        assert not jnp.allclose(new_params["latents"], current_params["latents"])
        # A should be unchanged
        assert jnp.allclose(
            new_params["flux"]["data"]["A"], current_params["flux"]["data"]["A"]
        )

    def test_optimize_block_numpyro_with_custom_optimizer(self, linear_model_and_data):
        """Test _optimize_block_numpyro with a custom optimizer."""
        model = linear_model_and_data["model"]
        data = linear_model_and_data["data"]
        true_A = linear_model_and_data["true_A"]
        true_latents = linear_model_and_data["true_latents"]

        current_params = {
            "latents": jnp.array(true_latents),
            "flux": {"data": {"A": jnp.array(true_A) + 0.1}, "err": {}},
        }

        block = ParameterBlock(
            name="flux",
            params="flux:data",
            optimizer=numpyro.optim.Adam,
            optimizer_kwargs={"step_size": 1e-2},
            num_steps=50,
        )

        new_params = _optimize_block_numpyro(
            model,
            data,
            block,
            current_params,
            rng_key=jax.random.PRNGKey(0),
        )

        assert "flux" in new_params
        assert "A" in new_params["flux"]["data"]


class TestOptimizeIterativeWithNonlinear:
    """Tests for optimize_iterative with non-linear transforms."""

    @pytest.fixture
    def nonlinear_model_and_data(self):
        """Create a model with non-linear FunctionTransform."""
        n_stars = 32
        n_latents = 4
        n_flux = 16

        rng = np.random.default_rng(42)

        # Simple nonlinear: y = sigmoid(latents @ A.T) * scale
        def sigmoid_transform(z, A, scale):
            return scale * jax.nn.sigmoid(z @ A.T)

        transform = FunctionTransform(
            output_size=n_flux,
            transform=jax.vmap(sigmoid_transform, in_axes=(0, None, None)),
            priors={
                "A": dist.Normal(0.0, 1.0),
                "scale": dist.HalfNormal(1.0),
            },
            shapes={
                "A": (n_flux, n_latents),
                "scale": (),
            },
            vmap=False,
        )

        model = plx.LVM(latent_size=n_latents)
        model.register_output("flux", transform)

        # Generate data
        true_A = rng.normal(size=(n_flux, n_latents)) * 0.5
        true_scale = 2.0
        true_latents = rng.normal(size=(n_stars, n_latents))
        true_flux = true_scale * 1.0 / (1.0 + np.exp(-(true_latents @ true_A.T)))
        flux_err = np.full_like(true_flux, 0.05)

        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                true_flux + rng.normal(0, flux_err),
                err=flux_err,
            ),
        )

        return model, data

    def test_optimize_iterative_nonlinear_runs(self, nonlinear_model_and_data):
        """Non-linear transforms fall back to SVI, and say so on the way."""
        model, data = nonlinear_model_and_data

        # Use default blocks (should auto-detect non-linear and use numpyro)
        with pytest.warns(
            PolluxLinearizationWarning, match="not affine in the latents"
        ):
            result = optimize_iterative(
                model,
                data,
                max_cycles=2,
                rng_key=jax.random.PRNGKey(42),
                progress=False,
            )

        assert isinstance(result, IterativeOptimizationResult)
        assert result.n_cycles >= 1
        assert len(result.losses_per_cycle) >= 1
        assert "latents" in result.params
        assert "flux" in result.params
        # Every block ran with SVI, and the result says which
        assert [b.optimizer for b in result.blocks] == [None, None]

    def test_the_fallback_warning_can_be_silenced(self, nonlinear_model_and_data):
        """The warning is a category, so the stdlib filter machinery turns it off."""
        model, data = nonlinear_model_and_data
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=PolluxLinearizationWarning)
            optimize_iterative(
                model,
                data,
                max_cycles=1,
                rng_key=jax.random.PRNGKey(42),
                progress=False,
            )

    def test_optimize_iterative_nonlinear_loss_decreases(
        self, nonlinear_model_and_data
    ):
        """Test that loss decreases during optimization."""
        model, data = nonlinear_model_and_data

        blocks = [
            ParameterBlock(
                name="latents",
                params="latents",
                num_steps=200,
            ),
            ParameterBlock(
                name="flux",
                params="flux:data",
                num_steps=200,
            ),
        ]

        result = optimize_iterative(
            model,
            data,
            blocks=blocks,
            max_cycles=3,
            rng_key=jax.random.PRNGKey(42),
            progress=False,
        )

        # Loss should generally decrease (or at least not increase much)
        assert result.losses_per_cycle[-1] <= result.losses_per_cycle[0] * 1.5

    def test_optimize_iterative_nonlinear_works_with_default_rng(
        self, nonlinear_model_and_data
    ):
        """Test that rng_key=None works (uses default key internally)."""
        model, data = nonlinear_model_and_data

        blocks = [
            ParameterBlock(
                name="flux",
                params="flux:data",
                num_steps=10,
            ),
        ]

        # Should work with rng_key=None - uses default key internally
        result = optimize_iterative(
            model,
            data,
            blocks=blocks,
            max_cycles=1,
            rng_key=None,
            progress=False,
        )
        assert isinstance(result, IterativeOptimizationResult)


class TestMixedLinearNonlinear:
    """Tests for optimize_iterative with mixed linear and non-linear outputs."""

    @pytest.fixture
    def mixed_model_and_data(self):
        """Create a model with both linear and non-linear outputs."""
        n_stars = 32
        n_latents = 4
        n_flux = 16
        n_labels = 2

        rng = np.random.default_rng(42)

        model = plx.LVM(latent_size=n_latents)

        # Linear output for labels
        model.register_output("label", LinearTransform(output_size=n_labels))

        # Non-linear output for flux
        def exp_transform(z, A):
            return jnp.exp(z @ A.T)

        transform = FunctionTransform(
            output_size=n_flux,
            transform=jax.vmap(exp_transform, in_axes=(0, None)),
            priors={"A": dist.Normal(0.0, 0.5)},
            shapes={"A": (n_flux, n_latents)},
            vmap=False,
        )
        model.register_output("flux", transform)

        # Generate data
        true_A_label = rng.normal(size=(n_labels, n_latents))
        true_A_flux = rng.normal(size=(n_flux, n_latents)) * 0.2
        true_latents = rng.normal(size=(n_stars, n_latents)) * 0.5

        true_label = true_latents @ true_A_label.T
        true_flux = np.exp(true_latents @ true_A_flux.T)

        data = plx.data.PolluxData(
            label=plx.data.OutputData(
                true_label + rng.normal(0, 0.1, size=true_label.shape),
                err=np.full_like(true_label, 0.1),
            ),
            flux=plx.data.OutputData(
                true_flux + rng.normal(0, 0.05, size=true_flux.shape),
                err=np.full_like(true_flux, 0.05),
            ),
        )

        return model, data

    def test_mixed_model_auto_blocks(self, mixed_model_and_data):
        """One non-linear output blocks the shared latents, but not the linear output."""
        model, data = mixed_model_and_data

        with pytest.warns(PolluxLinearizationWarning) as record:
            result = optimize_iterative(
                model,
                data,
                max_cycles=2,
                rng_key=jax.random.PRNGKey(42),
                progress=False,
            )

        assert isinstance(result, IterativeOptimizationResult)
        assert "latents" in result.params
        assert "label" in result.params
        assert "flux" in result.params

        # The latents couple every output, so one non-linear output rules them out;
        # the linear output's own parameters are still solved in closed form
        by_name = {b.name: b.optimizer for b in result.blocks}
        assert by_name["latents"] is None
        assert by_name["flux:data"] is None
        assert by_name["label:data"] == "least_squares"
        assert "2 of 3 blocks" in str(record[0].message)

    def test_mixed_model_explicit_blocks(self, mixed_model_and_data):
        """Test mixed model with explicit block specification."""
        model, data = mixed_model_and_data

        blocks = [
            # Use least squares for latents (only valid if all outputs linear,
            # so we use numpyro here)
            ParameterBlock(
                name="latents",
                params="latents",
                num_steps=100,
            ),
            # Use least squares for linear label output
            ParameterBlock(
                name="label",
                params="label:data",
                optimizer="least_squares",
            ),
            # Use numpyro for non-linear flux output
            ParameterBlock(
                name="flux",
                params="flux:data",
                num_steps=100,
            ),
        ]

        result = optimize_iterative(
            model,
            data,
            blocks=blocks,
            max_cycles=3,
            rng_key=jax.random.PRNGKey(42),
            progress=False,
        )

        assert result.n_cycles >= 1
        assert all(jnp.isfinite(loss) for loss in result.losses_per_cycle)


class TestPartialData:
    """A model is routinely applied to data holding only some of its outputs.

    Inferring labels from spectra alone is the standard test-set step. Every part of
    the fit has to agree to leave the absent outputs out -- the closed-form solves,
    the SVI blocks, the prior initialization, the loss and the default block list --
    and the ones that used to disagree failed looking for data or parameters that
    were never going to exist.
    """

    @pytest.fixture
    def model_and_partial_data(self):
        """Two parameterized linear outputs, but data for only one of them."""
        rng = np.random.default_rng(0)
        n_stars, n_labels, n_flux = 20, 3, 6

        model = plx.LVM(latent_size=2)
        model.register_output("label", LinearTransform(output_size=n_labels))
        model.register_output("flux", LinearTransform(output_size=n_flux))

        flux_only = plx.data.PolluxData(
            flux=plx.data.OutputData(
                jnp.array(rng.normal(size=(n_stars, n_flux))),
                err=jnp.full((n_stars, n_flux), 0.1),
            )
        )
        trained = {
            "latents": jnp.zeros((n_stars, 2)),
            "flux": {"data": {"A": jnp.array(rng.normal(size=(n_flux, 2)))}, "err": {}},
        }
        return model, flux_only, trained

    def test_participating_outputs(self, model_and_partial_data):
        model, data, _ = model_and_partial_data
        assert _participating_outputs(model, data) == ["flux"]

    def test_loss_ignores_absent_outputs(self, model_and_partial_data):
        """It used to predict every output first, needing parameters for the absent
        one before discarding its prediction."""
        model, data, trained = model_and_partial_data
        assert jnp.isfinite(_compute_loss(model, data, trained))

    def test_default_blocks_skip_absent_outputs(self, model_and_partial_data):
        """A block for an absent output reaches the solver and finds no data."""
        model, data, trained = model_and_partial_data
        result = optimize_iterative(
            model,
            data,
            max_cycles=1,
            rng_key=jax.random.PRNGKey(0),
            initial_params=trained,
            progress=False,
        )
        assert [b.name for b in result.blocks] == ["latents", "flux:data"]

    def test_prior_initialization_skips_absent_outputs(self, model_and_partial_data):
        """With neither fixed_pars nor initial_params, the parameters are drawn from
        the priors -- which ran the whole model over a dataset missing an output."""
        model, data, _ = model_and_partial_data
        result = optimize_iterative(
            model,
            data,
            blocks=["latents"],
            max_cycles=1,
            rng_key=jax.random.PRNGKey(0),
            progress=False,
        )
        assert result.params["latents"].shape == (len(data), model.latent_size)

    def test_end_to_end_label_inference(self, model_and_partial_data):
        """The whole point: fit the latents to a spectrum-only dataset."""
        model, data, trained = model_and_partial_data
        result = optimize_iterative(
            model,
            data,
            blocks=["latents"],
            fixed_pars=model.output_pars(trained),
            max_cycles=5,
            rng_key=jax.random.PRNGKey(0),
            progress=False,
        )
        latents = result.params["latents"]
        assert latents.shape == (len(data), model.latent_size)
        assert jnp.all(jnp.isfinite(latents))
