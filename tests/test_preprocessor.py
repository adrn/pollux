import numpy as np
import pytest

import pollux as plx


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def test_unit_shift_scale(rng):
    true_loc = [8.23, -1.33]
    true_scale = [4.74, 6.66]
    data = rng.normal(true_loc, true_scale, (8192, 2))

    # Test the basic initialization
    preprocessor = plx.data.ShiftScalePreprocessor(
        loc=np.array([0.0, 0.0]), scale=np.array([1.0, 1.0])
    )
    assert np.allclose(preprocessor.loc, np.array([0.0, 0.0]))
    assert np.allclose(preprocessor.scale, np.array([1.0, 1.0]))

    preprocessor = plx.data.ShiftScalePreprocessor.from_data(data)

    # Check that the loc and scale are computed correctly
    assert np.allclose(preprocessor.loc, true_loc, atol=0.2)
    assert np.allclose(preprocessor.scale, true_scale, atol=0.2)

    # Test the transformation
    transformed = preprocessor.transform(data)
    assert np.allclose(np.mean(transformed, axis=0), np.array([0.0, 0.0]), atol=1e-6)
    assert np.allclose(np.std(transformed, axis=0), np.array([1.0, 1.0]), atol=1e-6)

    # Test the inverse transformation
    reconstructed = preprocessor.inverse_transform(transformed)
    assert np.allclose(reconstructed, data, atol=1e-6)

    # Test from_data with axis=None - take mean/std over all elements
    preprocessor_all = plx.data.ShiftScalePreprocessor.from_data(data, axis=None)
    transformed_all = preprocessor_all.transform(data)
    assert np.allclose(np.mean(transformed_all), 0.0, atol=1e-6)
    assert np.allclose(np.std(transformed_all), 1.0, atol=1e-6)


def test_shift_scale_from_data_percentiles(rng):
    # Make data with some outliers
    true_loc = [8.23, -1.33]
    true_scale = [4.74, 6.66]
    data = rng.normal(true_loc, true_scale, (8192, 2))
    data = np.concatenate((data, rng.normal(0.0, 400.0, (16, 2))))

    # Test with default parameters (median, 16-84 percentiles)
    preprocessor = plx.data.ShiftScalePreprocessor.from_data_percentiles(data)
    assert np.allclose(preprocessor.loc, true_loc, atol=0.2)
    assert np.allclose(preprocessor.scale, true_scale, atol=0.3)

    # Test transform and inverse transform
    transformed = preprocessor.transform(data)
    reconstructed = preprocessor.inverse_transform(transformed)
    assert np.allclose(reconstructed, data, atol=1e-6)  # float32

    # Test with custom percentiles
    preprocessor_custom = plx.data.ShiftScalePreprocessor.from_data_percentiles(
        data, loc_percentile=25.0, scale_percentiles=(5.0, 95.0)
    )
    transformed_custom = preprocessor_custom.transform(data)
    reconstructed_custom = preprocessor_custom.inverse_transform(transformed_custom)
    assert np.allclose(reconstructed_custom, data, atol=1e-6)  # float32


def test_shift_scale_errors():
    # Test error transformation
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    errors = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    preprocessor = plx.data.ShiftScalePreprocessor.from_data(data)

    # Test error transformation
    transformed_errors = preprocessor.transform_err(errors)
    assert np.allclose(transformed_errors, errors / preprocessor.scale)

    # Test inverse error transformation
    reconstructed_errors = preprocessor.inverse_transform_err(transformed_errors)
    assert np.allclose(reconstructed_errors, errors)

    # Verify that errors scale correctly
    assert np.allclose(preprocessor.transform_err(errors) * preprocessor.scale, errors)


def test_shift_scale_with_nans():
    # Create test data with NaNs
    data = np.array([[1.0, 2.0], [3.0, float("nan")], [5.0, 6.0]])

    # Test from_data_percentiles handles NaNs correctly
    preprocessor = plx.data.ShiftScalePreprocessor.from_data_percentiles(data)

    # Transform data with NaNs
    transformed = preprocessor.transform(data)

    # Check that NaNs remain NaNs
    assert np.isnan(transformed[1, 1])

    # Check that non-NaN values are transformed correctly
    valid_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    valid_transformed = preprocessor.transform(valid_data)

    # The column with NaN should still have reasonable transformations for non-NaN values
    assert not np.any(np.isnan(valid_transformed))
