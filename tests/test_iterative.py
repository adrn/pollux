"""Tests for iterative optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux.models.iterative import (
    IterativeOptimizationResult,
    ParameterBlock,
    _get_regularization_from_prior,
    _is_linear_transform,
    _solve_latents_least_squares,
    _solve_output_params_least_squares,
    optimize_iterative,
)
from pollux.models.transforms import (
    LinearTransform,
    TransformSequence,
)

jax.config.update("jax_enable_x64", True)


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


class TestIsLinearTransform:
    """Tests for _is_linear_transform utility."""

    def test_linear_transform_is_linear(self):
        trans = LinearTransform(output_size=8)
        assert _is_linear_transform(trans) is True

    def test_sequence_not_supported(self):
        trans = TransformSequence(
            transforms=(
                LinearTransform(output_size=8),
                LinearTransform(output_size=4),
            )
        )
        # TransformSequence is not supported for iterative optimization
        assert _is_linear_transform(trans) is False


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


class TestLeastSquaresSolvers:
    """Tests for the least squares solvers."""

    @pytest.fixture
    def simple_linear_model(self):
        """Create a simple linear model for testing."""
        n_stars = 64
        n_latents = 8
        n_flux = 32

        model = plx.LuxModel(latent_size=n_latents)
        model.register_output("flux", LinearTransform(output_size=n_flux))

        # Generate some synthetic data
        rng = np.random.default_rng(42)
        true_A = rng.normal(size=(n_flux, n_latents))
        true_latents = rng.normal(size=(n_stars, n_latents))
        true_flux = true_latents @ true_A.T
        flux_err = np.abs(rng.normal(0.1, 0.02, size=true_flux.shape))

        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                true_flux + rng.normal(0, flux_err),
                err=flux_err,
            ),
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

    def test_solve_latents_shape(self, simple_linear_model):
        """Test that latents solver returns correct shape."""
        model = simple_linear_model["model"]
        data = simple_linear_model["data"]
        true_A = simple_linear_model["true_A"]
        n_stars = simple_linear_model["n_stars"]
        n_latents = simple_linear_model["n_latents"]

        current_params = {
            "flux": {"data": {"A": jnp.array(true_A)}, "err": {}},
        }

        latents = _solve_latents_least_squares(model, data, current_params)
        assert latents.shape == (n_stars, n_latents)

    def test_solve_output_params_shape(self, simple_linear_model):
        """Test that output params solver returns correct shape."""
        model = simple_linear_model["model"]
        data = simple_linear_model["data"]
        true_latents = simple_linear_model["true_latents"]
        n_flux = simple_linear_model["n_flux"]
        n_latents = simple_linear_model["n_latents"]

        output_params = _solve_output_params_least_squares(
            model, data, "flux", jnp.array(true_latents)
        )

        assert "A" in output_params
        assert output_params["A"].shape == (n_flux, n_latents)


class TestOptimizeIterative:
    """Tests for the main optimize_iterative function."""

    @pytest.fixture
    def simple_model_and_data(self):
        """Create a simple model and data for testing."""
        n_stars = 32
        n_latents = 4
        n_flux = 16

        rng = np.random.default_rng(123)

        model = plx.LuxModel(latent_size=n_latents)
        model.register_output("flux", LinearTransform(output_size=n_flux))

        # Generate data
        true_A = rng.normal(size=(n_flux, n_latents)) * 0.5
        true_latents = rng.normal(size=(n_stars, n_latents))
        true_flux = true_latents @ true_A.T
        flux_err = np.full_like(true_flux, 0.1)

        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                true_flux + rng.normal(0, flux_err),
                err=flux_err,
            ),
        )

        return model, data, true_A, true_latents

    def test_optimize_iterative_basic(self, simple_model_and_data):
        """Test basic iterative optimization."""
        model, data, _, _ = simple_model_and_data

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

    def test_optimize_iterative_with_custom_blocks(self, simple_model_and_data):
        """Test with custom parameter blocks."""
        model, data, _, _ = simple_model_and_data

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

    def test_optimize_iterative_convergence(self, simple_model_and_data):
        """Test that optimization can converge."""
        model, data, _, _ = simple_model_and_data

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

    def test_optimize_iterative_with_initial_params(self, simple_model_and_data):
        """Test optimization with initial parameters."""
        model, data, true_A, true_latents = simple_model_and_data

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
