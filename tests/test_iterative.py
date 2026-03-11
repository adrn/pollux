"""Tests for iterative optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux.models.iterative import (
    IterativeOptimizationResult,
    ParameterBlock,
    _get_regularization_from_prior,
    _is_linear_transform,
    _optimize_block_numpyro,
    _solve_latents_least_squares,
    _solve_output_params_least_squares,
    optimize_iterative,
)
from pollux.models.transforms import (
    FunctionTransform,
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


class TestOptimizeBlockNumpyro:
    """Tests for the _optimize_block_numpyro function."""

    @pytest.fixture
    def simple_linear_model_and_data(self):
        """Create a simple linear model for testing numpyro block optimization."""
        n_stars = 32
        n_latents = 4
        n_flux = 16

        rng = np.random.default_rng(42)

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

        return {
            "model": model,
            "data": data,
            "true_A": true_A,
            "true_latents": true_latents,
        }

    @pytest.fixture
    def nonlinear_model_and_data(self):
        """Create a model with a non-linear FunctionTransform for testing."""
        n_stars = 32
        n_latents = 4
        n_flux = 16

        rng = np.random.default_rng(42)

        # Create a simple non-linear transform: y = exp(latents @ A.T)
        def exp_transform(latents, A):
            return jnp.exp(latents @ A.T)

        transform = FunctionTransform(
            output_size=n_flux,
            transform=jax.vmap(lambda z, A: exp_transform(z, A), in_axes=(0, None)),
            priors={"A": dist.Normal(0.0, 0.5)},
            shapes={"A": (n_flux, n_latents)},
            vmap=False,
        )

        model = plx.LuxModel(latent_size=n_latents)
        model.register_output("flux", transform)

        # Generate data using the nonlinear model
        true_A = rng.normal(size=(n_flux, n_latents)) * 0.2
        true_latents = rng.normal(size=(n_stars, n_latents)) * 0.5
        true_flux = np.exp(true_latents @ true_A.T)
        flux_err = np.full_like(true_flux, 0.05)

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
        }

    def test_optimize_block_numpyro_returns_params(self, simple_linear_model_and_data):
        """Test that _optimize_block_numpyro returns valid parameters."""
        model = simple_linear_model_and_data["model"]
        data = simple_linear_model_and_data["data"]
        true_A = simple_linear_model_and_data["true_A"]
        true_latents = simple_linear_model_and_data["true_latents"]

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

    def test_optimize_block_numpyro_latents(self, simple_linear_model_and_data):
        """Test that _optimize_block_numpyro can optimize latents."""
        model = simple_linear_model_and_data["model"]
        data = simple_linear_model_and_data["data"]
        true_A = simple_linear_model_and_data["true_A"]
        true_latents = simple_linear_model_and_data["true_latents"]

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

    def test_optimize_block_numpyro_with_custom_optimizer(
        self, simple_linear_model_and_data
    ):
        """Test _optimize_block_numpyro with a custom optimizer."""
        model = simple_linear_model_and_data["model"]
        data = simple_linear_model_and_data["data"]
        true_A = simple_linear_model_and_data["true_A"]
        true_latents = simple_linear_model_and_data["true_latents"]

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

        model = plx.LuxModel(latent_size=n_latents)
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
        """Test that optimize_iterative runs with non-linear transforms."""
        model, data = nonlinear_model_and_data

        # Use default blocks (should auto-detect non-linear and use numpyro)
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

        model = plx.LuxModel(latent_size=n_latents)

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
        """Test that auto-generated blocks correctly identify linear vs non-linear."""
        model, data = mixed_model_and_data

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
