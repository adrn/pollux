from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..typing import BatchedDataT


class AbstractPreprocessor(eqx.Module):
    """Base class for data preprocessors."""

    @classmethod
    @abstractmethod
    def from_data(cls, data: BatchedDataT) -> "AbstractPreprocessor":
        """Compute preprocessing parameters from data."""

    @abstractmethod
    def transform(self, X: BatchedDataT) -> BatchedDataT:
        """Apply preprocessing transform to the input data."""

    @abstractmethod
    def inverse_transform(self, X: BatchedDataT) -> BatchedDataT:
        """Apply inverse preprocessing transform to the input data."""

    @abstractmethod
    def transform_err(self, X_err: BatchedDataT) -> BatchedDataT:
        """Apply preprocessing transform to the input data uncertainties."""

    @abstractmethod
    def inverse_transform_err(self, X_err: BatchedDataT) -> BatchedDataT:
        """Apply inverse preprocessing transform to the input data uncertainties."""

    def __call__(self, X: BatchedDataT, inverse: bool = False) -> BatchedDataT:
        """Apply preprocessing transform or inverse to the input data."""
        if inverse:
            return self.inverse_transform(X)
        return self.transform(X)


class NullPreprocessor(AbstractPreprocessor):
    """A preprocessor that does nothing to data.

    Examples
    --------
    This data preprocessor does nothing to the  centers the data on the mean and scales to unit
    variance, both computed along axis=0::

    >>> import numpy as np
    >>> from pollux.data import NormalizePreprocessor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(1.0, 2.0, size=(1024, 10))
    >>> preprocessor = NormalizePreprocessor()
    >>> preprocessor.fit(X)
    >>> assert np.allclose(preprocessor.loc, np.mean(X, axis=0))
    >>> assert np.allclose(preprocessor.scale, np.std(X, axis=0))
    """

    @classmethod
    def from_data(cls, *_: Any) -> "NullPreprocessor":
        """Compute preprocessing parameters from data."""
        return cls()

    def transform(self, X: BatchedDataT) -> BatchedDataT:
        """Apply preprocessing transform to the input data."""
        return X

    def inverse_transform(self, X: BatchedDataT) -> BatchedDataT:
        """Apply inverse preprocessing transform to the input data."""
        return X

    def transform_err(self, X_err: BatchedDataT) -> BatchedDataT:
        """Apply preprocessing transform to the input data uncertainties."""
        return X_err

    def inverse_transform_err(self, X_err: BatchedDataT) -> BatchedDataT:
        """Apply inverse preprocessing transform to the input data uncertainties."""
        return X_err


class ShiftScalePreprocessor(AbstractPreprocessor):
    """Shift and then scale the data.

    The data are shifted by the specified location parameter `loc` and then scaled by
    the `scale` parameter.

    Use the ``from_data()`` and ``from_data_percentiles()`` class methods to compute the
    preprocessing parameters from specified data. The ``from_data()`` method computes
    the mean and standard deviation of the data along the specified axis, while the
    ``from_data_percentiles()`` method computes the median and the difference between
    the specified percentiles as the scale.

    Examples
    --------
    The default way of computing the preprocessing parameters uses the ``from_data()``
    class method, which computes the mean and standard deviation of the data along
    axis=0::

    >>> import jax.numpy as jnp
    >>> data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> from pollux.data import ShiftScalePreprocessor
    >>> preprocessor = ShiftScalePreprocessor.from_data(data)
    >>> processed_data = preprocessor.transform(data)
    >>> assert jnp.allclose(jnp.mean(processed_data, axis=0), 0.0)
    >>> assert jnp.allclose(jnp.std(processed_data, axis=0), 1.0)

    To instead use the mean and standard deviation computed over all axes at the same
    time, set the axis to None::

    >>> preprocessor = ShiftScalePreprocessor.from_data(data, axis=None)
    >>> processed_data = preprocessor.transform(data)
    >>> assert jnp.allclose(jnp.mean(processed_data, axis=0), jnp.mean(processed_data))
    >>> assert jnp.allclose(jnp.std(processed_data, axis=0), jnp.std(processed_data))

    An alternative way of computing the preprocessing parameters uses the
    ``from_data_percentiles()`` class method, which computes the median and the
    difference between the specified percentiles as the scale. Here we will specify
    using (1/2 times) the difference of the 84th and 16th percentile values as the
    scale::

    >>> preprocessor = ShiftScalePreprocessor.from_data_percentiles(data, 16.0, 84.0)
    >>> processed_data = preprocessor.transform(data)
    >>> assert jnp.allclose(jnp.median(processed_data, axis=0), 0.0)


    """

    loc: jax.Array = eqx.field(converter=jnp.asarray)
    scale: jax.Array = eqx.field(converter=jnp.asarray)

    @classmethod
    def from_data(cls, data: BatchedDataT, axis: int = 0) -> "ShiftScalePreprocessor":
        """Compute preprocessing parameters from data.

        Parameters
        ----------
        data
            The data to preprocess.
        axis
            The axis along which to compute the mean and standard deviation.
        """
        return cls(jnp.mean(data, axis=axis), jnp.std(data, axis=axis))

    @classmethod
    def from_data_percentiles(
        cls,
        data: BatchedDataT,
        percentile_low: float = 16.0,
        percentile_high: float = 84.0,
        axis: int = 0,
    ) -> "ShiftScalePreprocessor":
        """Compute preprocessing parameters from data.

        Parameters
        ----------
        data
            The data to preprocess.
        percentile_low
            The lower percentile to use for computing the scale.
        percentile_high
            The higher / upper percentile to use for computing the scale.
        axis
            The axis along which to compute the mean and standard deviation.
        """
        _scale = (
            jnp.diff(
                jnp.percentile(
                    data,
                    jnp.array([percentile_low, percentile_high]),
                    axis=axis,
                ),
                axis=0,
            )
            / 2.0
        )
        return cls(jnp.median(data, axis=axis), _scale)

    def transform(self, X: BatchedDataT) -> BatchedDataT:
        """Apply preprocessing transform to the input data."""
        return (X - self.loc) / self.scale

    def inverse_transform(self, X: BatchedDataT) -> BatchedDataT:
        """Apply inverse preprocessing transform to the input data."""
        return X * self.scale + self.loc

    def transform_err(self, X_err: BatchedDataT) -> BatchedDataT:
        """Apply preprocessing transform to the input data uncertainties."""
        return X_err / self.scale

    def inverse_transform_err(self, X_err: BatchedDataT) -> BatchedDataT:
        """Apply inverse preprocessing transform to the input data uncertainties."""
        return X_err * self.scale
