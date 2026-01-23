"""Tests for the Cannon model."""

import jax.numpy as jnp
import numpy as np
import pytest

import pollux as plx
from pollux.models import Cannon
from pollux.models.transforms import LinearTransform, PolyFeatureTransform


class TestCannonBasic:
    """Basic tests for Cannon initialization and properties."""

    def test_init(self):
        """Test basic initialization."""
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
        assert cannon.label_size == 3
        assert cannon.output_size == 100
        assert cannon.poly_degree == 2
        assert cannon.include_bias is True
        assert cannon.is_fitted is False
        assert cannon.coeffs is None
        assert cannon.scatter is None

    def test_n_features(self):
        """Test n_features property for various configurations."""
        # 3 labels, degree 2, with bias: C(3+2, 2) = 10
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
        assert cannon.n_features == 10

        # 2 labels, degree 2, with bias: C(2+2, 2) = 6
        cannon = Cannon(label_size=2, output_size=100, poly_degree=2)
        assert cannon.n_features == 6

        # 3 labels, degree 1, with bias: C(3+1, 1) = 4
        cannon = Cannon(label_size=3, output_size=100, poly_degree=1)
        assert cannon.n_features == 4

        # 3 labels, degree 2, no bias: C(3+2, 2) - 1 = 9
        cannon = Cannon(
            label_size=3, output_size=100, poly_degree=2, include_bias=False
        )
        assert cannon.n_features == 9

    def test_get_features(self):
        """Test polynomial feature expansion."""
        cannon = Cannon(label_size=2, output_size=10, poly_degree=2)
        labels = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        features = cannon.get_features(labels)

        assert features.shape == (2, 6)
        # For degree 2 with bias: [1, x1, x2, x1^2, x1*x2, x2^2]
        expected_row_0 = jnp.array([1.0, 1.0, 2.0, 1.0, 2.0, 4.0])
        assert jnp.allclose(features[0], expected_row_0)


class TestCannonFit:
    """Tests for Cannon fitting."""

    def test_fit_simple_linear(self):
        """Test fitting a simple linear relationship (degree 1)."""
        rng = np.random.default_rng(42)
        n_stars = 100
        n_labels = 2
        n_pixels = 50

        # Generate labels
        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))

        # Create true coefficients: bias + linear terms
        # For degree=1: features are [1, x1, x2], so 3 features
        true_coeffs = jnp.array(rng.standard_normal((n_pixels, 3)))

        # Generate features and output
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=1)
        features = cannon.get_features(labels)
        spectra = features @ true_coeffs.T

        # Fit
        cannon = cannon.fit(labels, spectra)

        assert cannon.is_fitted
        assert cannon.coeffs.shape == (n_pixels, 3)
        assert cannon.scatter.shape == (n_pixels,)

        # Coefficients should match well (no noise in data)
        assert jnp.allclose(cannon.coeffs, true_coeffs, atol=1e-5)

    def test_fit_with_noise(self):
        """Test fitting with noisy data."""
        rng = np.random.default_rng(42)
        n_stars = 500
        n_labels = 3
        n_pixels = 100
        noise_level = 0.1

        # Generate labels
        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))

        # Create true coefficients
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=2)
        true_coeffs = jnp.array(rng.standard_normal((n_pixels, cannon.n_features)))

        # Generate output with noise
        features = cannon.get_features(labels)
        spectra_true = features @ true_coeffs.T
        noise = jnp.array(rng.standard_normal((n_stars, n_pixels)) * noise_level)
        spectra = spectra_true + noise

        # Fit with inverse variance
        ivar = jnp.ones_like(spectra) / noise_level**2
        cannon = cannon.fit(labels, spectra, output_ivar=ivar)

        assert cannon.is_fitted
        # Scatter should be close to noise level
        assert jnp.isclose(jnp.mean(cannon.scatter), noise_level, atol=0.02)

    def test_fit_with_regularization(self):
        """Test that regularization shrinks coefficients."""
        rng = np.random.default_rng(42)
        n_stars = 100
        n_labels = 3
        n_pixels = 50

        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))
        spectra = jnp.array(rng.standard_normal((n_stars, n_pixels)))

        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=2)

        # Fit without regularization
        cannon_unreg = cannon.fit(labels, spectra, regularization=0.0)

        # Fit with regularization
        cannon_reg = cannon.fit(labels, spectra, regularization=1.0)

        # Regularized coefficients should have smaller norm
        unreg_norm = jnp.linalg.norm(cannon_unreg.coeffs)
        reg_norm = jnp.linalg.norm(cannon_reg.coeffs)
        assert reg_norm < unreg_norm

    def test_fit_validation_errors(self):
        """Test that fit raises errors for invalid inputs."""
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)

        # Wrong label size
        with pytest.raises(ValueError, match="Expected labels with 3 columns"):
            cannon.fit(jnp.zeros((10, 2)), jnp.zeros((10, 100)))

        # Wrong output size
        with pytest.raises(ValueError, match="Expected output with 100 columns"):
            cannon.fit(jnp.zeros((10, 3)), jnp.zeros((10, 50)))

        # Mismatched number of stars
        with pytest.raises(ValueError, match="labels has 10 stars but output has 20"):
            cannon.fit(jnp.zeros((10, 3)), jnp.zeros((20, 100)))


class TestCannonPredict:
    """Tests for Cannon prediction."""

    def test_predict_unfitted_raises(self):
        """Test that predict raises error before fitting."""
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cannon.predict(jnp.zeros((10, 3)))

    def test_predict_wrong_label_size(self):
        """Test that predict raises error for wrong label size."""
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
        # Manually set coeffs to make it "fitted"
        cannon = Cannon(
            label_size=3,
            output_size=100,
            poly_degree=2,
            coeffs=jnp.zeros((100, 10)),
            scatter=jnp.zeros(100),
        )
        with pytest.raises(ValueError, match="Expected labels with 3 columns"):
            cannon.predict(jnp.zeros((10, 2)))

    def test_predict_roundtrip(self):
        """Test that fit/predict roundtrips work."""
        rng = np.random.default_rng(42)
        n_stars = 100
        n_labels = 3
        n_pixels = 50

        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=2)
        true_coeffs = jnp.array(rng.standard_normal((n_pixels, cannon.n_features)))

        features = cannon.get_features(labels)
        spectra = features @ true_coeffs.T

        # Fit and predict
        cannon = cannon.fit(labels, spectra)
        predicted = cannon.predict(labels)

        assert jnp.allclose(predicted, spectra, atol=1e-5)


class TestCannonTransformSequence:
    """Tests for Cannon-to-TransformSequence conversion."""

    def test_to_transform_sequence(self):
        """Test conversion to TransformSequence."""
        cannon = Cannon(label_size=3, output_size=128, poly_degree=2)
        transform = cannon.to_transform_sequence()

        assert isinstance(transform, plx.models.TransformSequence)
        assert len(transform.transforms) == 2
        assert isinstance(transform.transforms[0], PolyFeatureTransform)
        assert isinstance(transform.transforms[1], LinearTransform)
        assert transform.transforms[0].degree == 2
        assert transform.transforms[1].output_size == 128

    def test_get_coeffs_unfitted_raises(self):
        """Test that getting coeffs before fitting raises error."""
        cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            cannon.get_coeffs_as_transform_pars()

    def test_get_coeffs_as_transform_pars(self):
        """Test getting coefficients in transform parameter format."""
        rng = np.random.default_rng(42)
        n_stars = 100
        n_labels = 3
        n_pixels = 50

        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=2)

        features = cannon.get_features(labels)
        spectra = jnp.array(rng.standard_normal((n_stars, n_pixels)))

        cannon = cannon.fit(labels, spectra)
        pars = cannon.get_coeffs_as_transform_pars()

        assert "data" in pars
        assert isinstance(pars["data"], list)
        assert len(pars["data"]) == 2
        assert pars["data"][0] == {}  # PolyFeatureTransform has no params
        assert "A" in pars["data"][1]
        assert pars["data"][1]["A"].shape == (n_pixels, cannon.n_features)

    def test_cannon_with_lux_model(self):
        """Test using Cannon's TransformSequence with LuxModel."""
        rng = np.random.default_rng(42)
        n_stars = 50
        n_labels = 3
        n_pixels = 20

        # Create and fit Cannon
        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=2)
        features = cannon.get_features(labels)
        true_coeffs = jnp.array(rng.standard_normal((n_pixels, cannon.n_features)))
        spectra = features @ true_coeffs.T

        cannon = cannon.fit(labels, spectra)

        # Create LuxModel with Cannon's transform
        model = plx.LuxModel(latent_size=n_labels)
        transform = cannon.to_transform_sequence()
        model.register_output("flux", transform)

        # Get Cannon's fitted params and use them to predict
        cannon_pars = cannon.get_coeffs_as_transform_pars()

        # Predict using LuxModel with Cannon's coefficients
        predicted = model.predict_outputs(labels, {"flux": cannon_pars})

        # Should match Cannon's prediction
        cannon_predicted = cannon.predict(labels)
        assert jnp.allclose(predicted["flux"], cannon_predicted, atol=1e-5)


class TestCannonEndToEnd:
    """End-to-end integration tests for Cannon."""

    def test_realistic_workflow(self):
        """Test a realistic Cannon workflow with train/test split."""
        rng = np.random.default_rng(42)
        n_train = 500
        n_test = 100
        n_labels = 3
        n_pixels = 100
        noise_level = 0.05

        # Generate training data
        train_labels = jnp.array(rng.standard_normal((n_train, n_labels)))

        # True model: polynomial relationship
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=2)
        true_coeffs = jnp.array(rng.standard_normal((n_pixels, cannon.n_features)))
        true_coeffs = true_coeffs * 0.1  # Scale down for realistic magnitudes

        train_features = cannon.get_features(train_labels)
        train_spectra_true = train_features @ true_coeffs.T
        train_noise = jnp.array(rng.standard_normal((n_train, n_pixels)) * noise_level)
        train_spectra = train_spectra_true + train_noise
        train_ivar = jnp.ones_like(train_spectra) / noise_level**2

        # Generate test data
        test_labels = jnp.array(rng.standard_normal((n_test, n_labels)))
        test_features = cannon.get_features(test_labels)
        test_spectra_true = test_features @ true_coeffs.T

        # Fit Cannon
        cannon = cannon.fit(train_labels, train_spectra, train_ivar)

        # Predict on test set
        predicted = cannon.predict(test_labels)

        # Check prediction accuracy
        residuals = predicted - test_spectra_true
        rms = jnp.sqrt(jnp.mean(residuals**2))

        # Should be close to noise level (can be somewhat larger due to finite
        # training set and regularization)
        assert rms < noise_level * 2

    def test_high_degree_polynomial(self):
        """Test with higher degree polynomial."""
        rng = np.random.default_rng(42)
        n_stars = 200
        n_labels = 2
        n_pixels = 30

        labels = jnp.array(rng.standard_normal((n_stars, n_labels)))
        cannon = Cannon(label_size=n_labels, output_size=n_pixels, poly_degree=4)

        # With degree 4 and 2 labels: C(2+4, 4) = 15 features
        assert cannon.n_features == 15

        features = cannon.get_features(labels)
        true_coeffs = jnp.array(rng.standard_normal((n_pixels, cannon.n_features)))
        spectra = features @ true_coeffs.T

        cannon = cannon.fit(labels, spectra)
        predicted = cannon.predict(labels)

        # Higher degree polynomials have more numerical precision issues
        assert jnp.allclose(predicted, spectra, atol=1e-3)
