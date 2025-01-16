from typing import TYPE_CHECKING

import equinox as eqx
import jax
from jax.typing import ArrayLike

from .preprocessor import AbstractPreprocessor, NullPreprocessor


class OutputData(eqx.Module):
    """A container for observed output data.

    This class is used to store a single block of observed output data, such as fluxes
    for a collection of stars, or stellar labels for a set of stars, or other data like
    broadband fluxes, etc. Each instance of this class should correspond to a single
    output data type (e.g., spectral fluxes should be a separate instance from stellar
    labels).

    Parameters
    ----------
    data : array-like
        The observed output data.
    err : array-like, optional
        The errors on the observed output data.
    preprocessor : AbstractPreprocessor, optional
        A preprocessor to apply to the data. For example, this might center on the mean
        and scale to unit variance (using the ``NormalizePreprocessor``).
    processed : array-like, optional
        The preprocessed data. This is set to None by default and is updated when the
        ``.preprocess()`` method is called.

    Examples
    --------
    Let's assume you have a set of spectra for a collection of 128 stars. The data are
    aligned on the same wavelength grid with 2048 pixels. The data can therefore be
    stored in a 2D array with shape (128, 2048). You also have the errors on the fluxes,
    which are stored in a 2D array with the same shape. You can create an instance of
    ``OutputData`` to store this data. In the example below, we will generate some
    random data to represent this case for the sake of illustration and will initially
    use a no-op data preprocessor, so the pre-processed data should be equal to the
    input data::

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
        preprocessor=NullPreprocessor(),
        data_processed=f32[128,2048],
        err_processed=f32[128,2048]
    )
    >>> assert np.all(flux_data.data_processed == flux_data.data)
    >>> assert np.all(flux_data.err_processed == flux_data.err)

    Instead, we could specify a data pre-processor to rescale and center the input data.
    For this, we use the ``NormalizePreprocessor``, which centers the data on the mean
    and scales to unit variance::

    >>> from pollux.data import NormalizePreprocessor
    >>> flux_data = OutputData(
    ...     data=spectra, err=spectra_err, preprocessor=NormalizePreprocessor()
    ... )
    >>> assert np.allclose(np.std(flux_data.data_processed, axis=0), 1.0)

    """

    data: jax.Array = eqx.field(converter=jax.numpy.asarray)
    err: jax.Array | None = eqx.field(
        default=None,
        converter=lambda x: jax.numpy.asarray(x) if x is not None else None,
    )
    preprocessor: AbstractPreprocessor = eqx.field(default=NullPreprocessor())
    data_processed: jax.Array | None = eqx.field(default=None)
    err_processed: jax.Array | None = eqx.field(default=None)

    def __post_init__(self) -> None:
        if self.err is not None and self.data.shape != self.err.shape:
            msg = "Data and error arrays must have the same shape"
            raise ValueError(msg)
        self._preprocess()

    def _preprocess(self) -> None:
        self.preprocessor.fit(self.data)
        self.data_processed = self.preprocessor.transform(self.data)
        # TODO: need to also process the errors

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice | ArrayLike) -> "OutputData":
        """Get a slice of the data.

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
        return type(self)(
            data=sliced_data,
            err=sliced_err,
            preprocessor=self.preprocessor,
            data_processed=None
            if self.data_processed is None
            else self.data_processed[key],
            err_processed=None
            if self.err_processed is None
            else self.err_processed[key],
        )


class PolluxData(dict[str, OutputData]):
    def __init__(self, **kwargs: OutputData) -> None:
        """A data container for storing observed outputs from a Pollux model.

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

    def __len__(self) -> int:
        return len(next(iter(self.values())).data)
