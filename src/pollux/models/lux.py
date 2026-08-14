"""Lux: a multi-output latent variable model for spectra and labels.

Lux (Horta et al. 2025) represents each object by a latent vector and generates every
observed output as a linear function of the latent vector. Because the spectra and the
labels are outputs of the same latents, a model trained on stars with both can infer
labels for stars with only a spectrum, and vice versa.

References
----------
Horta, D. R., Price-Whelan, A. M., Hogg, D. W., Ness, M., Casey, A. R. 2025, arXiv:2502.01745
"""

from collections.abc import Mapping, Sequence

from .lvm import LVM, select_outputs
from .transforms import LinearTransform, scatter_at_scale

__all__ = ["Lux"]


class Lux(LVM):
    """The Lux model: stellar labels and spectra, both linear in the latents.

    Lux is a model for stellar spectroscopy specifically, so it registers exactly two
    outputs and names them itself: ``"label"`` and ``"flux"``. Both are a
    :class:`~pollux.models.transforms.LinearTransform` of the latent vectors, and
    either can carry a fitted intrinsic scatter
    (:class:`~pollux.models.transforms.ScatterTransform`) on its reported errors.

    Because both outputs are generated from the *same* latents, a model trained on
    stars that have both can infer labels for stars with only a spectrum, and vice
    versa.

    The :class:`~pollux.data.PolluxData` you fit must therefore be keyed ``"label"``
    and ``"flux"`` to match. For different names, a different number of outputs, or
    anything else the two-output structure cannot express, build the model with
    :class:`~pollux.models.LVM` and :meth:`~pollux.models.LVM.register_output`
    directly.

    Lux is a linear model. For a polynomial relationship between the labels and the
    spectrum, see :class:`~pollux.models.Cannon`.

    Parameters
    ----------
    latent_size
        The size of the latent vector representation for each star. Typically larger
        than ``label_size`` and much smaller than ``flux_size``.
    label_size
        Number of stellar labels (e.g. Teff, logg, [Fe/H]), registered as ``"label"``.
    flux_size
        Number of spectral pixels, registered as ``"flux"``.
    intrinsic_scatter
        Which outputs get a fitted per-element scatter added in quadrature to their
        reported errors. By default (``None``) that is the flux and not the labels: a
        spectrum's uncertainties are typically under-reported, through telluric
        residuals, sky lines or simply bad pixels, in a way catalog label errors are
        not. Pass ``True`` for both, ``False`` for neither, ``"label"``/``"flux"`` or a
        sequence of them, or a mapping from output name to the scale of the
        ``HalfNormal`` prior on the scatter. That scale depends on how the output was
        preprocessed, so pass a mapping when the default of 1.0 is a poor match.

    Examples
    --------
    A model for 6 stellar labels and 1000 spectral pixels sharing a 16-dimensional
    latent space:

    >>> import pollux as plx
    >>> model = plx.Lux(latent_size=16, label_size=6, flux_size=1000)
    >>> sorted(model.outputs)
    ['flux', 'label']
    >>> model.latent_size
    16

    By default the flux has a scatter to fit and the labels do not:

    >>> type(model.outputs["flux"].err_transform).__name__
    'ScatterTransform'
    >>> type(model.outputs["label"].err_transform).__name__
    'NoOpTransform'

    Widen the prior on that scatter to suit the preprocessing, or ask for one on the
    labels too:

    >>> model = plx.Lux(
    ...     latent_size=16,
    ...     label_size=6,
    ...     flux_size=1000,
    ...     intrinsic_scatter={"label": 0.1, "flux": 5.0},
    ... )
    >>> float(model.outputs["flux"].err_transform.priors["s"].scale)
    5.0

    A Lux model is an :class:`~pollux.models.LVM`, so it is fitted and used the same way:

    >>> isinstance(model, plx.LVM)
    True
    """

    _LABEL_NAME = "label"
    _FLUX_NAME = "flux"

    def __init__(
        self,
        latent_size: int,
        label_size: int,
        flux_size: int,
        intrinsic_scatter: bool | Sequence[str] | Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(latent_size=latent_size)

        # The spectrum's uncertainties are the ones typically under-reported; the
        # catalog label errors are taken at face value unless asked otherwise
        if intrinsic_scatter is None:
            intrinsic_scatter = [self._FLUX_NAME]

        sizes = {self._LABEL_NAME: label_size, self._FLUX_NAME: flux_size}
        scatter_scales = select_outputs(intrinsic_scatter, sizes, "intrinsic_scatter")

        for name, size in sizes.items():
            self.register_output(
                name,
                LinearTransform(output_size=size),
                err_transform=(
                    scatter_at_scale(size, scatter_scales[name])
                    if name in scatter_scales
                    else None
                ),
            )
