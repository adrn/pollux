__all__ = [
    "AbstractPreprocessor",
    "NormalizePreprocessor",
    "NullPreprocessor",
    "PercentilePreprocessor",
]

from abc import abstractmethod
from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractPreprocessor(eqx.Module):
    """Base class for data preprocessors."""

    def _validate_parameters(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) is None:
                msg = (
                    "All preprocessor parameters must be set (missing value for: "
                    f"{field.name})"
                )
                raise ValueError(msg)

    @abstractmethod
    def fit(self, X: jax.Array) -> None:
        """Compute internal transform parameters from the input data."""

    @abstractmethod
    def transform(self, X: jax.Array) -> jax.Array:
        """Apply preprocessing to data."""

    @abstractmethod
    def inverse_transform(self, X: jax.Array) -> jax.Array:
        """Reverse preprocessing."""

    @abstractmethod
    def transform_err(self, X_err: jax.Array) -> jax.Array:
        """Apply preprocessing to uncertainty."""

    @abstractmethod
    def inverse_transform_err(self, X_err: jax.Array) -> jax.Array:
        """Reverse preprocessing to uncertainty."""

    def __call__(self, X: jax.Array, inverse: bool = False) -> jax.Array:
        if inverse:
            return self.inverse_transform(X)
        return self.transform(X)


class NullPreprocessor(AbstractPreprocessor):
    """A preprocessor that does nothing to data."""

    def fit(self, X: jax.Array) -> None:
        pass

    def transform(self, X: jax.Array) -> jax.Array:
        return X

    def inverse_transform(self, X: jax.Array) -> jax.Array:
        return X

    def transform_err(self, X_err: jax.Array) -> jax.Array:
        return X_err

    def inverse_transform_err(self, X_err: jax.Array) -> jax.Array:
        return X_err


class NormalizePreprocessor(AbstractPreprocessor):
    """Recenter on the mean and scale to unit variance."""

    loc: jax.Array | None = None
    scale: jax.Array | None = None

    def fit(self, X: jax.Array) -> None:
        self.loc = jnp.mean(X, axis=0)
        self.scale = jnp.std(X, axis=0)

    def transform(self, X: jax.Array) -> jax.Array:
        self._validate_parameters()
        return (X - self.loc) / self.scale

    def inverse_transform(self, X: jax.Array) -> jax.Array:
        self._validate_parameters()
        return X * self.scale + self.loc

    def transform_err(self, X_err: jax.Array) -> jax.Array:
        self._validate_parameters()
        return X_err / self.scale

    def inverse_transform_err(self, X_err: jax.Array) -> jax.Array:
        self._validate_parameters()
        return X_err * self.scale


class PercentilePreprocessor(NormalizePreprocessor):
    """Recenter on the median and scale to a robust estimate of the variance."""

    percentile_low: float = 16.0
    percentile_high: float = 84.0

    def fit(self, X: jax.Array) -> None:
        self.loc = jnp.median(X, axis=0)
        self.scale = (
            jnp.diff(
                jnp.percentile(
                    X, jnp.array([self.percentile_low, self.percentile_high]), axis=0
                ),
                axis=0,
            )
            / 2.0
        )
