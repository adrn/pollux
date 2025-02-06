import jax
import jax.numpy as jnp
import pytest

from pollux.data import NullPreprocessor, OutputData, ShiftScalePreprocessor


@pytest.fixture(scope="class")
def sample_arrays():
    rngs = jax.random.split(jax.random.PRNGKey(0), 4)
    return {
        "flux": jax.random.normal(rngs[0], (100, 10)),
        "flux_err": jax.random.uniform(
            rngs[1], shape=(100, 10), minval=1.0, maxval=2.0
        ),
        "label": jax.random.normal(rngs[2], (100, 3)),
        "label_err": jax.random.uniform(
            rngs[3], shape=(100, 3), minval=1.0, maxval=2.0
        ),
    }


class TestOutputData:
    def test_creation(self, sample_arrays):
        col = OutputData(data=sample_arrays["flux"])
        assert len(col) == 100
        assert col.data.shape == (100, 10)
        assert col.err is None
        assert col.processed is False

    def test_creation_with_errors(self, sample_arrays):
        col = OutputData(data=sample_arrays["flux"], err=sample_arrays["flux_err"])
        assert col.err.shape == col.data.shape  # type: ignore[union-attr]
        assert col.processed is False

    def test_shape_mismatch(self, sample_arrays):
        with pytest.raises(
            ValueError, match="Data and error arrays must have the same shape"
        ):
            OutputData(data=sample_arrays["flux"], err=sample_arrays["label_err"])

    def test_preprocess_default(self, sample_arrays):
        col = OutputData(data=sample_arrays["flux"])
        new_col = col.preprocess()
        assert new_col.processed is True
        assert isinstance(col.preprocessor, NullPreprocessor)
        assert jnp.all(new_col.data == col.data)

    def test_preprocess_custom(self, sample_arrays):
        col = OutputData(
            data=sample_arrays["flux"],
            err=sample_arrays["flux_err"],
            preprocessor=ShiftScalePreprocessor(1.0, 2.0),  # type: ignore[arg-type]
        )
        new_col = col.preprocess()
        assert new_col.processed is True
        assert isinstance(col.preprocessor, ShiftScalePreprocessor)
        assert jnp.allclose(new_col.data, (col.data - 1.0) / 2.0)

        roundtrip = new_col.unprocess()
        assert jnp.allclose(roundtrip.data, col.data, atol=1e-4)
        assert roundtrip.processed is False

        sub_col = col[:10]
        assert len(sub_col) == 10
        assert sub_col.preprocessor is col.preprocessor

        new_sub_col = sub_col.preprocess()
        assert jnp.allclose(new_sub_col.data, new_col.data[:10])
