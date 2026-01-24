"""The Cannon: a data-driven model for stellar spectra.

The Cannon (Ness et al. 2015) learns a polynomial relationship between stellar
labels (like Teff, logg, [Fe/H]) and spectra. Given reference stars with known
labels and spectra, it fits a per-pixel polynomial model that can then predict
spectra for new labels or infer labels from new spectra.

This implementation provides:
- A simple `fit()`/`predict()` interface for the traditional Cannon workflow
- Integration with the pollux transform system via `PolyFeatureTransform` + `LinearTransform`

References
----------
Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015, ApJ, 808, 16

Examples
--------
>>> import jax.numpy as jnp
>>> from pollux.models import Cannon

Create and fit a Cannon model:

>>> labels = jnp.array([[5000., 4.0, -0.5], [5500., 3.5, 0.0]])  # doctest: +SKIP
>>> spectra = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # doctest: +SKIP
>>> cannon = Cannon(label_size=3, output_size=2, poly_degree=2)  # doctest: +SKIP
>>> cannon = cannon.fit(labels, spectra)  # doctest: +SKIP

Predict spectra for new labels:

>>> new_labels = jnp.array([[5200., 3.8, -0.2]])  # doctest: +SKIP
>>> predicted = cannon.predict(new_labels)  # doctest: +SKIP
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .transforms import (
    LinearTransform,
    PolyFeatureTransform,
    TransformSequence,
    _compute_n_poly_features,
)

__all__ = ["Cannon"]


class Cannon(eqx.Module):
    """The Cannon: a data-driven model for stellar spectra.

    Learns a polynomial relationship between stellar labels and spectra (or other
    outputs). For each output pixel, the model fits:

        output_λ = Σ_j θ_{λj} · feature_j(labels)

    where the features are polynomial combinations of the labels up to the
    specified degree.

    Parameters
    ----------
    label_size
        Number of input labels (e.g., 3 for Teff, logg, [Fe/H]).
    output_size
        Number of output dimensions (e.g., number of spectral pixels).
    poly_degree
        Maximum polynomial degree for feature expansion. Default is 2.
    include_bias
        Whether to include a bias term in the polynomial features. Default is True.

    Attributes
    ----------
    coeffs
        Fitted coefficients, shape ``(output_size, n_features)``. None before fitting.
    scatter
        Fitted per-pixel scatter, shape ``(output_size,)``. None before fitting.
    n_features
        Number of polynomial features (computed from label_size and poly_degree).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from pollux.models import Cannon

    Create a Cannon model for 3 labels and 100 spectral pixels:

    >>> cannon = Cannon(label_size=3, output_size=100, poly_degree=2)
    >>> cannon.n_features  # 1 + 3 + 6 = 10 for degree 2 with 3 labels
    10

    The number of features follows the formula for combinations with replacement:
    C(n_labels + degree, degree) = C(3 + 2, 2) = 10
    """

    label_size: int
    output_size: int
    poly_degree: int = 2
    include_bias: bool = True

    # Fitted parameters (None before fitting)
    coeffs: jax.Array | None = eqx.field(default=None, repr=False)
    scatter: jax.Array | None = eqx.field(default=None, repr=False)

    # Internal transform (initialized in __post_init__)
    _poly_transform: PolyFeatureTransform = eqx.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the polynomial feature transform."""
        poly_transform = PolyFeatureTransform(
            degree=self.poly_degree, include_bias=self.include_bias
        )
        object.__setattr__(self, "_poly_transform", poly_transform)

    @property
    def n_features(self) -> int:
        """Number of polynomial features."""
        return _compute_n_poly_features(
            self.label_size, self.poly_degree, self.include_bias
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self.coeffs is not None

    def get_features(self, labels: jax.Array) -> jax.Array:
        """Expand labels into polynomial features.

        Parameters
        ----------
        labels
            Stellar labels, shape ``(n_stars, label_size)``.

        Returns
        -------
        array
            Polynomial features, shape ``(n_stars, n_features)``.

        Examples
        --------
        >>> labels = jnp.array([[1.0, 2.0]])  # 1 star, 2 labels
        >>> cannon = Cannon(label_size=2, output_size=10, poly_degree=2)
        >>> features = cannon.get_features(labels)
        >>> features.shape
        (1, 6)
        >>> features  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        Array([[1., 1., 2., 1., 2., 4.]], dtype=float...)
        """
        return self._poly_transform.apply(labels)

    def fit(
        self,
        labels: jax.Array,
        output: jax.Array,
        output_ivar: jax.Array | None = None,
        regularization: float = 0.0,
    ) -> Cannon:
        """Fit the Cannon using weighted least squares.

        For each output pixel, solves the weighted least squares problem:

            argmin_θ Σ_i w_i (y_i - f_i @ θ)^2 + λ ||θ||^2

        Parameters
        ----------
        labels
            Training stellar labels, shape ``(n_stars, label_size)``.
        output
            Training output (e.g., spectra), shape ``(n_stars, output_size)``.
        output_ivar
            Inverse variance of the output. Shape ``(n_stars, output_size)``.
            If None, uniform weights (1.0) are used.
        regularization
            L2 regularization strength (λ). Default is 0.0 (no regularization).
            Larger values shrink coefficients toward zero.

        Returns
        -------
        Cannon
            A new Cannon instance with fitted coefficients and scatter.

        Notes
        -----
        This method uses JAX's vmap for efficient vectorized fitting across all
        pixels. The solution for each pixel is:

            θ = (F.T @ W @ F + λI)^{-1} @ F.T @ W @ y

        where F is the design matrix (polynomial features), W is a diagonal
        weight matrix, and y is the output vector.

        Examples
        --------
        >>> labels = jnp.array([[5000., 4.0], [5500., 3.5], [6000., 4.5]])
        >>> spectra = jnp.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
        >>> cannon = Cannon(label_size=2, output_size=2, poly_degree=1)
        >>> cannon = cannon.fit(labels, spectra)
        >>> cannon.is_fitted
        True
        """
        n_stars = labels.shape[0]

        # Validate shapes
        if labels.shape[1] != self.label_size:
            msg = (
                f"Expected labels with {self.label_size} columns, got {labels.shape[1]}"
            )
            raise ValueError(msg)
        if output.shape[0] != n_stars:
            msg = f"labels has {n_stars} stars but output has {output.shape[0]} stars"
            raise ValueError(msg)
        if output.shape[1] != self.output_size:
            msg = (
                f"Expected output with {self.output_size} columns, "
                f"got {output.shape[1]}"
            )
            raise ValueError(msg)

        # Expand labels to polynomial features
        features = self._poly_transform.apply(labels)  # (n_stars, n_features)

        if output_ivar is None:
            output_ivar = jnp.ones_like(output)

        # Regularization matrix
        reg_matrix = regularization * jnp.eye(self.n_features)

        def fit_single_pixel(
            pixel_data: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            """Fit a single pixel using weighted least squares."""
            y, w = pixel_data  # output and inverse variance for this pixel

            # Weighted normal equations: (F.T @ W @ F + λI) @ θ = F.T @ W @ y
            # where W = diag(w)
            FtW = features.T * w  # (n_features, n_stars) - broadcasting
            FtWF = FtW @ features  # (n_features, n_features)
            FtWy = FtW @ y  # (n_features,)

            # Solve with regularization
            theta = jnp.linalg.solve(FtWF + reg_matrix, FtWy)

            # Compute scatter (weighted RMS of residuals)
            pred = features @ theta
            residuals = y - pred
            weighted_ss = jnp.sum(w * residuals**2)
            sum_weights = jnp.sum(w)
            scatter_val = jnp.sqrt(weighted_ss / sum_weights)

            return theta, scatter_val

        # Vectorize over pixels (axis 1 of output and output_ivar)
        # We need to transpose so pixels are the leading dimension
        pixel_data = (output.T, output_ivar.T)  # (output_size, n_stars) each
        coeffs, scatter = jax.vmap(fit_single_pixel)(pixel_data)

        # coeffs shape: (output_size, n_features)
        # scatter shape: (output_size,)

        new_self: Cannon = eqx.tree_at(
            lambda c: (c.coeffs, c.scatter),
            self,
            (coeffs, scatter),
            is_leaf=lambda x: x is None,
        )
        return new_self

    def predict(self, labels: jax.Array) -> jax.Array:
        """Predict output for given labels.

        Parameters
        ----------
        labels
            Stellar labels, shape ``(n_stars, label_size)``.

        Returns
        -------
        array
            Predicted output, shape ``(n_stars, output_size)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Examples
        --------
        >>> cannon = cannon.fit(train_labels, train_spectra)  # doctest: +SKIP
        >>> predicted = cannon.predict(test_labels)  # doctest: +SKIP
        """
        if not self.is_fitted:
            msg = "Cannon must be fitted before prediction. Call fit() first."
            raise RuntimeError(msg)

        if labels.shape[1] != self.label_size:
            msg = (
                f"Expected labels with {self.label_size} columns, got {labels.shape[1]}"
            )
            raise ValueError(msg)

        features = self._poly_transform.apply(labels)  # (n_stars, n_features)
        assert self.coeffs is not None  # Guaranteed by is_fitted check above
        return features @ self.coeffs.T  # (n_stars, output_size)

    def to_transform_sequence(self) -> TransformSequence:
        """Convert to a TransformSequence for use with LuxModel.

        Returns a TransformSequence that can be used with LuxModel for Bayesian
        inference or more complex models. The sequence consists of:

        1. PolyFeatureTransform: labels → polynomial features (no learnable params)
        2. LinearTransform: features → output (learnable A matrix)

        Returns
        -------
        TransformSequence
            A transform sequence that can be registered with LuxModel.

        Notes
        -----
        This method creates a new TransformSequence where the LinearTransform's
        A matrix will be sampled from priors during numpyro inference. If the
        Cannon has been fitted, you can use the fitted coefficients as initial
        values or fixed parameters.

        Examples
        --------
        >>> import pollux as plx
        >>> cannon = Cannon(label_size=3, output_size=128, poly_degree=2)
        >>> transform = cannon.to_transform_sequence()
        >>> model = plx.LuxModel(latent_size=3)  # latent_size = label_size
        >>> model.register_output("flux", transform)
        """
        return TransformSequence(
            transforms=(  # type: ignore[arg-type]
                PolyFeatureTransform(
                    degree=self.poly_degree, include_bias=self.include_bias
                ),
                LinearTransform(output_size=self.output_size),
            )
        )

    def get_coeffs_as_transform_pars(
        self,
    ) -> dict[str, list[dict[str, jax.Array | None]]]:
        """Get fitted coefficients in transform parameter format.

        Returns the fitted coefficients in the format expected by
        TransformSequence/LuxModel. This allows using Cannon-fitted parameters
        as initial values or fixed parameters in LuxModel.

        Returns
        -------
        dict
            Parameter dictionary in the format:
            ``{"data": [{"A": coeffs.T}]}``

            The coefficients are transposed because LinearTransform expects
            shape ``(output_size, latent_size)`` where latent_size = n_features.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Examples
        --------
        >>> cannon = cannon.fit(labels, spectra)  # doctest: +SKIP
        >>> pars = cannon.get_coeffs_as_transform_pars()  # doctest: +SKIP
        >>> # Use with LuxModel
        >>> model.predict_outputs(labels, {"flux": pars})  # doctest: +SKIP
        """
        if not self.is_fitted:
            msg = "Cannon must be fitted to get coefficients."
            raise RuntimeError(msg)

        # PolyFeatureTransform has no parameters (index 0)
        # LinearTransform has A matrix (index 1)
        # TransformSequence expects a list of dicts
        return {"data": [{}, {"A": self.coeffs}]}
