import jax
import jax.numpy as jnp
import pytest

import pollux as plx
from pollux.data import NullPreprocessor, OutputData, PolluxData, ShiftScalePreprocessor
from pollux.data.data import warn_if_unprocessed
from pollux.exceptions import PolluxPreprocessingWarning
from pollux.models.transforms import LinearTransform


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
        assert col.err.shape == ()
        assert col.err == jnp.array(0.0)
        assert col.processed is False

    def test_creation_with_errors(self, sample_arrays):
        col = OutputData(data=sample_arrays["flux"], err=sample_arrays["flux_err"])
        assert col.err.shape == col.data.shape
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


class TestPolluxData:
    def test_creation(self, sample_arrays):
        ddata = {
            "flux": OutputData(
                data=sample_arrays["flux"], err=sample_arrays["flux_err"]
            ),
            "label": OutputData(
                data=sample_arrays["label"], err=sample_arrays["label_err"]
            ),
        }
        data = PolluxData(**ddata)
        assert len(data) == 100
        assert set(data.keys()) == {"flux", "label"}
        assert isinstance(data["flux"], OutputData)
        assert isinstance(data["label"], OutputData)

    def test_invalid_output_type(self, sample_arrays):
        data = {
            "flux": OutputData(data=sample_arrays["flux"]),
            "label": sample_arrays["label"],  # Not an OutputData instance
        }
        with pytest.raises(
            ValueError, match="Output data must be instances of OutputData"
        ):
            PolluxData(**data)

    def test_length_mismatch(self, sample_arrays):
        data = {
            "flux": OutputData(data=sample_arrays["flux"][:50]),  # First 50 rows
            "label": OutputData(data=sample_arrays["label"]),  # All rows
        }
        with pytest.raises(
            ValueError, match="All output data must have the same length"
        ):
            PolluxData(**data)

    def test_slicing(self, sample_arrays):
        ddata = {
            "flux": OutputData(data=sample_arrays["flux"]),
            "label": OutputData(data=sample_arrays["label"]),
        }
        data = PolluxData(**ddata)

        sliced = data[:10]
        assert len(sliced) == 10
        assert isinstance(sliced, PolluxData)
        assert set(sliced.keys()) == {"flux", "label"}

    def test_preprocessing(self, sample_arrays):
        ddata = {
            "flux": OutputData(
                data=sample_arrays["flux"],
                preprocessor=ShiftScalePreprocessor(1.0, 2.0),  # type: ignore[arg-type]
            ),
            "label": OutputData(
                data=sample_arrays["label"],
                preprocessor=ShiftScalePreprocessor(0.0, 1.0),  # type: ignore[arg-type]
            ),
        }
        data = PolluxData(**ddata)

        processed = data.preprocess()
        assert all(v.processed for v in processed.values())
        assert all(not v.processed for v in data.values())

        unprocessed = processed.unprocess()
        assert all(not v.processed for v in unprocessed.values())
        assert jnp.allclose(unprocessed["flux"].data, data["flux"].data, atol=1e-4)
        assert jnp.allclose(unprocessed["label"].data, data["label"].data, atol=1e-4)


class TestWarnIfUnprocessed:
    """The guard against fitting data that carries a preprocessor it never went through.

    See :func:`pollux.data.data.warn_if_unprocessed`.
    """

    def _data(self, sample_arrays, preprocessor):
        return PolluxData(
            flux=OutputData(data=sample_arrays["flux"], preprocessor=preprocessor),
            label=OutputData(data=sample_arrays["label"]),
        )

    def test_warns_when_unprocessed(self, sample_arrays):
        data = self._data(sample_arrays, ShiftScalePreprocessor(1.0, 2.0))  # type: ignore[arg-type]
        with pytest.warns(PolluxPreprocessingWarning, match=r"\['flux'\]"):
            warn_if_unprocessed(data, "optimize()")

    def test_silent_once_preprocessed(self, sample_arrays):
        data = self._data(sample_arrays, ShiftScalePreprocessor(1.0, 2.0))  # type: ignore[arg-type]
        # pytest is configured to turn warnings into errors, so a warning here fails
        warn_if_unprocessed(data.preprocess(), "optimize()")

    def test_silent_for_null_preprocessor(self, sample_arrays):
        warn_if_unprocessed(self._data(sample_arrays, NullPreprocessor()), "optimize()")

    def test_optimize_warns(self, sample_arrays):
        """The guard is wired into the fitting entry points, not just available."""
        model = plx.LVM(latent_size=2)
        model.register_output("flux", LinearTransform(output_size=10))
        data = PolluxData(
            flux=OutputData(
                data=sample_arrays["flux"],
                err=sample_arrays["flux_err"],
                preprocessor=ShiftScalePreprocessor.from_data(sample_arrays["flux"]),
            )
        )

        with pytest.warns(PolluxPreprocessingWarning):
            model.optimize(data, num_steps=1, rng_key=jax.random.PRNGKey(0))

        with pytest.warns(PolluxPreprocessingWarning):
            model.optimize_iterative(data, max_cycles=1, progress=False)
