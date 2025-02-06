__all__ = ["OutputData", "PolluxData"]

from typing import TYPE_CHECKING, Any, Union

import equinox as eqx
import jax.numpy as jnp
from dataclassish.converters import Optional
from jax.typing import ArrayLike
from xmmutablemap import ImmutableMap

from ..typing import BatchedDataT
from .preprocessor import AbstractPreprocessor, NullPreprocessor


class OutputData(eqx.Module):
    """A container for single block of output data.

    This class is used to store data for a single output of a model, such as fluxes for
    a collection of stars, or stellar labels for a set of stars, or other data like
    broadband magnitudes, etc. Each instance of this class should correspond to a single
    output data type (e.g., spectral fluxes should be a separate instance from stellar
    labels).

    Parameters
    ----------
    data : array-like
        The output data.
    err : array-like, optional
        The uncertainties (errors) of the output data.
    preprocessor : AbstractPreprocessor, optional
        A preprocessor to apply to the data. For example, this might recenter the data
        on the mean and scale to unit variance (using the ``NormalizePreprocessor``).
        Use the ``.processed`` attribute to check if an instance has already been
        preprocessed.

    Examples
    --------
    Let's assume you have a set of spectra for a collection of 128 stars. The data are
    aligned on the same wavelength grid with 2048 pixels. The data can therefore be
    stored in a 2D array with shape (128, 2048). You also have the errors on the fluxes,
    which are stored in a 2D array with the same shape. You can create an instance of
    ``OutputData`` to store this data. In the example below, we will generate some
    random data to represent this case (for the sake of illustration)::

    >>> import numpy as np
    >>> from pollux.data import OutputData
    >>> rng = np.random.default_rng(seed=42)
    >>> spectra = rng.uniform(0, 10, size=(128, 2048))
    >>> spectra_err = rng.uniform(0, 1, size=spectra.shape)
    >>> flux_data = OutputData(data=spectra, err=spectra_err)
    >>> flux_data
    OutputData(
        data=f32[128,2048],
        err=f32[128,2048],
        preprocessor=NullPreprocessor
    )
    >>> assert flux_data.processed == False

    We did not specify a preprocessor, so the data are not preprocessed even if we call
    ``.preprocess()``. In this case, the processed data should equal the unprocessed
    data::

    >>> tmp = flux_data.preprocess()
    >>> assert tmp.processed == False
    >>> assert np.all(tmp.data, flux_data.data)

    We can instead specify a data preprocessor to rescale and center the input data. For
    this, we use the ``NormalizePreprocessor``, which centers the data on the mean and
    scales the data to unit variance, by default along ``axis=0``::

    >>> from pollux.data import NormalizePreprocessor
    >>> flux_data = OutputData(
    ...     data=spectra, err=spectra_err, preprocessor=NormalizePreprocessor
    ... )
    >>> processed_data = flux_data.preprocess()
    >>> assert processed_data.processed
    >>> assert np.allclose(np.mean(processed_data.data, axis=0), 0.0)
    >>> assert np.allclose(np.std(processed_data.data, axis=0), 1.0)

    """

    data: BatchedDataT = eqx.field(converter=jnp.asarray)
    err: BatchedDataT | None = eqx.field(default=None, converter=Optional(jnp.asarray))
    preprocessor: AbstractPreprocessor = eqx.field(default=NullPreprocessor())
    processed: bool = eqx.field(default=False)

    def __post_init__(self) -> None:
        """Validate the input data."""
        if self.err is not None and self.data.shape != self.err.shape:
            msg = "Data and error arrays must have the same shape"
            raise ValueError(msg)

    def preprocess(self) -> "OutputData":
        """Preprocess the data using the preprocessor."""
        if self.processed:
            return self

        return OutputData(
            data=self.preprocessor.transform(self.data),
            err=self.preprocessor.transform_err(self.err)
            if self.err is not None
            else None,
            preprocessor=self.preprocessor,
            processed=True,
        )

    def unprocess(self, data: BatchedDataT | None = None) -> "OutputData":
        """Unprocess the data using the preprocessor.

        Parameters
        ----------
        data
            The data to unprocess. If None, the instance's data will be unprocessed.
        """
        if data is not None and not self.processed:
            msg = "Data is not processed, so it cannot be unprocessed"
            raise ValueError(msg)

        if data is None:
            data = self.data

        return OutputData(
            data=self.preprocessor.inverse_transform(data),
            err=self.preprocessor.inverse_transform_err(data)
            if self.err is not None
            else None,
            preprocessor=self.preprocessor,
            processed=False,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice | ArrayLike) -> "OutputData":
        """Get a slice of the data. This preserves the preprocessor instance.

        Parameters
        ----------
        key : int, slice, or array-like
            The indexing key

        Returns
        -------
        OutputData
            A new OutputData instance with the sliced data
        """
        sliced_data = self.data[key]
        sliced_err = None if self.err is None else self.err[key]
        return OutputData(
            data=sliced_data, err=sliced_err, preprocessor=self.preprocessor
        )


class PolluxData(ImmutableMap[str, OutputData]):  # type: ignore[misc]
    def __init__(self, **kwargs: OutputData) -> None:
        """A data container for observed outputs from a Pollux model.

        For example, this could be the observed fluxes, labels, and errors for a set of
        stars.

        Parameters
        ----------
        **kwargs
            The output data to store. Each keyword argument should be a string key
            corresponding to the name of the output data, and the value should be an
            OutputData instance.

        """
        super().__init__(**kwargs)
        self._validate()

    def __getitem__(
        self, key: int | slice | ArrayLike | str
    ) -> Union["OutputData", Any]:
        if isinstance(key, str):
            return super().__getitem__(key)
        return self.__class__(**{name: output[key] for name, output in self.items()})

    def _validate(self) -> None:
        for name, output in self.items():
            if not TYPE_CHECKING and not isinstance(output, OutputData):
                msg = (
                    f"Output data must be instances of OutputData, not {type(output)} "
                    f"(invalid key: {name})"
                )
                raise ValueError(msg)

            if len(output) != len(self):
                msg = (
                    "All output data must have the same length along axis=0 (invalid "
                    f"key: {name})"
                )
                raise ValueError(msg)

    def preprocess(self) -> "PolluxData":
        """Preprocess all output data."""
        return self.__class__(
            **{name: output.preprocess() for name, output in self.items()}
        )

    def unprocess(
        self, data: Union["PolluxData", dict[str, BatchedDataT], None] = None
    ) -> "PolluxData":
        """Unprocess all output data."""
        data = data or self

        if set(self.keys()) != set(data.keys()):
            msg = "Data to unprocess must have the same keys as the instance"
            raise ValueError(msg)

        return self.__class__(
            **{name: self[name].unprocess(output) for name, output in data.items()}
        )

    def __len__(self) -> int:
        return len(next(iter(self.values())).data)
