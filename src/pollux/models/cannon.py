"""The Cannon: a data-driven model for stellar spectra.

The Cannon (Ness et al. 2015) learns a polynomial relationship between stellar labels
(like Teff, logg, [Fe/H]) and spectra. Given reference stars with known labels and
spectra, it fits a per-pixel polynomial model that can then predict spectra for new
labels, or infer labels for new spectra.

In this package the Cannon is an architecture built on :class:`~pollux.models.LVM`: the
latent vectors *are* the stellar labels, observed through a
:class:`~pollux.models.transforms.NoOpTransform`, and the spectrum is generated from
them by a polynomial feature expansion followed by a linear map. Training and label
inference are then the framework's ordinary fitting methods, and the per-pixel
coefficient solve is the closed-form weighted least squares that
:func:`~pollux.models.optimize_iterative` already recognizes.

References
----------
Ness, M., Hogg, D. W., Rix, H.-W., Ho, A. Y. Q., & Zasowski, G. 2015, ApJ, 808, 16
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import numpyro.distributions as dist

from .lvm import LVM, select_outputs
from .transforms import (
    LinearTransform,
    NoOpTransform,
    PolyFeatureTransform,
    ScatterTransform,
    TransformSequence,
    _compute_n_poly_features,
    polynomial_features,
)

__all__ = ["Cannon"]


class Cannon(LVM):
    """The Cannon: a data-driven model for stellar spectra.

    For each output element (e.g. each spectral pixel) the model is

    .. math::

        y_\\lambda = \\sum_j \\theta_{\\lambda j} \\, f_j(\\ell)

    where :math:`\\ell` are the stellar labels and :math:`f_j` are the polynomial
    combinations of them up to ``poly_degree``. The coefficients :math:`\\theta` appear
    in the parameter dictionary as the ``"A"`` matrix of the output's linear transform.

    Two outputs are registered: the labels, observed directly through a
    :class:`~pollux.models.transforms.NoOpTransform` (which is what makes the latents
    the labels), and the spectrum, generated through a
    :class:`~pollux.models.transforms.PolyFeatureTransform` followed by a
    :class:`~pollux.models.transforms.LinearTransform`.

    Parameters
    ----------
    label_size
        Number of stellar labels (e.g. 3 for Teff, logg, [Fe/H]). This is also the
        model's ``latent_size``.
    output_size
        Number of output elements (e.g. the number of spectral pixels).
    poly_degree
        Maximum polynomial degree of the label feature expansion. Default is 2.
    include_bias
        Whether the polynomial features include a constant term. Default is True.
    label_name, output_name
        The output names to register, which should match the keys of the
        :class:`~pollux.data.PolluxData` you intend to fit.
    intrinsic_scatter
        Which outputs get a fitted per-element scatter added in quadrature to their
        reported errors: ``True`` for both (default), ``False`` for neither, a sequence
        of output names, or a mapping from output name to the scale of the
        ``HalfNormal`` prior on the scatter. The scatter on the spectrum is the
        Cannon's per-pixel :math:`s_\\lambda`.
    coeff_prior
        Prior on the polynomial coefficients. Regularization enters here: a narrower
        prior shrinks the coefficients harder, and
        :func:`~pollux.models.optimize_iterative` turns it into the ridge term of the
        closed-form solve. Defaults to the standard unit Gaussian.

    Attributes
    ----------
    n_features
        Number of polynomial features, computed from ``label_size``, ``poly_degree``
        and ``include_bias``.

    Examples
    --------
    A Cannon for 3 labels and 100 spectral pixels:

    >>> import pollux as plx
    >>> cannon = plx.Cannon(label_size=3, output_size=100, poly_degree=2)
    >>> cannon.n_features  # 1 + 3 + 6 = 10 for degree 2 with 3 labels
    10
    >>> sorted(cannon.outputs)
    ['flux', 'label']
    >>> cannon.latent_size  # the latents are the labels
    3

    The feature count is the number of combinations with replacement,
    ``C(n_labels + degree, degree) = C(3 + 2, 2) = 10``.

    Training is the classic Cannon step: pin the latents to the observed labels and
    solve the coefficients exactly, which costs one linear solve.

    >>> res = cannon.optimize_iterative(  # doctest: +SKIP
    ...     train_data,
    ...     blocks=["flux:data"],
    ...     fixed_pars={"latents": train_data["label"].data},
    ...     max_cycles=1,
    ... )

    From there you can let the latents and the scatters float too, by running the
    default blocks from that starting point with ``initial_params=res.params``.

    To infer labels for stars with only a spectrum, optimize the latents with the
    trained output parameters held fixed:

    >>> test_res = cannon.optimize_iterative(  # doctest: +SKIP
    ...     test_data,
    ...     blocks=["latents"],
    ...     fixed_pars=cannon.output_pars(res.params),
    ... )
    >>> labels = test_res.params["latents"]  # doctest: +SKIP

    A Cannon is an :class:`~pollux.models.LVM`, so it is fitted and used the same way:

    >>> isinstance(cannon, plx.LVM)
    True

    Notes
    -----
    Solving for the labels is not a closed-form problem here, and
    :func:`~pollux.models.optimize_iterative` will say so: with ``poly_degree`` above 1
    the spectrum is not affine in the labels, so any block containing ``"latents"`` is
    downgraded from a linear solve to gradient descent. That warning is expected for a
    Cannon, not a sign of a misspecified model. Fitting the *coefficients* stays exact,
    because they enter linearly.

    The same non-convexity means label inference has more than one local optimum. Most
    stars land in the right one from a cold start, but a minority will not, so pass
    ``initial_params`` with a reasonable first guess at the labels when accuracy for
    every star matters.
    """

    poly_degree: int = 2
    include_bias: bool = True

    def __init__(
        self,
        label_size: int,
        output_size: int,
        poly_degree: int = 2,
        include_bias: bool = True,
        label_name: str = "label",
        output_name: str = "flux",
        intrinsic_scatter: bool | Sequence[str] | Mapping[str, float] = True,
        coeff_prior: dist.Distribution | None = None,
    ) -> None:
        super().__init__(latent_size=label_size)
        self.poly_degree = poly_degree
        self.include_bias = include_bias

        if label_name == output_name:
            msg = (
                f"label_name and output_name are both '{label_name}', but the Cannon "
                "registers them as two separate outputs."
            )
            raise ValueError(msg)

        scatter_scales = select_outputs(
            intrinsic_scatter, [label_name, output_name], "intrinsic_scatter"
        )

        def scatter_for(name: str, size: int) -> ScatterTransform | None:
            if name not in scatter_scales:
                return None
            scale = scatter_scales[name]
            # No scale given means keep the transform's own default prior
            if scale is None:
                return ScatterTransform(output_size=size)
            return ScatterTransform(
                output_size=size, priors={"s": dist.HalfNormal(scale)}
            )

        # The labels are the latents, observed directly. optimize_iterative detects
        # this passthrough and warm-starts the latents from the observed labels.
        self.register_output(
            label_name,
            NoOpTransform(output_size=label_size),
            err_transform=scatter_for(label_name, label_size),
        )

        linear = (
            LinearTransform(output_size=output_size)
            if coeff_prior is None
            else LinearTransform(output_size=output_size, priors={"A": coeff_prior})
        )
        self.register_output(
            output_name,
            TransformSequence(
                transforms=(
                    PolyFeatureTransform(degree=poly_degree, include_bias=include_bias),
                    linear,
                )
            ),
            err_transform=scatter_for(output_name, output_size),
        )

    @property
    def n_features(self) -> int:
        """Number of polynomial features."""
        return _compute_n_poly_features(
            self.latent_size, self.poly_degree, self.include_bias
        )

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
        >>> import jax.numpy as jnp
        >>> import pollux as plx
        >>> labels = jnp.array([[1.0, 2.0]])  # 1 star, 2 labels
        >>> cannon = plx.Cannon(label_size=2, output_size=10, poly_degree=2)
        >>> features = cannon.get_features(labels)
        >>> features.shape
        (1, 6)
        >>> features  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        Array([[1., 1., 2., 1., 2., 4.]], dtype=float...)
        """
        return polynomial_features(labels, self.poly_degree, self.include_bias)
