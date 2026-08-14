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

import numpyro.distributions as dist

from .lvm import LVM, select_outputs
from .transforms import (
    AbstractSingleTransform,
    LinearTransform,
    QuadraticTransform,
    ScatterTransform,
)

__all__ = ["Lux"]


class Lux(LVM):
    """The Lux model: outputs generated linearly (or quadratically) from the latents.

    Each named output gets a :class:`~pollux.models.transforms.LinearTransform` from the
    latent vectors, or a :class:`~pollux.models.transforms.QuadraticTransform` where
    ``quadratic`` selects it, and a fitted intrinsic scatter
    (:class:`~pollux.models.transforms.ScatterTransform`) on its reported errors where
    ``intrinsic_scatter`` selects it.

    Parameters
    ----------
    latent_size
        The size of the latent vector representation for each object.
    outputs
        The outputs to register, mapping each output name to its size. The names should
        match the keys of the :class:`~pollux.data.PolluxData` you intend to fit.
    quadratic
        Which outputs are quadratic rather than linear in the latents: ``True`` for all
        of them, ``False`` for none (default), or a sequence of output names. A
        quadratic output has ``latent_size**2`` more parameters per output element, so
        this is usually worth selecting per output rather than globally.
    intrinsic_scatter
        Which outputs get a fitted per-element scatter added in quadrature to their
        reported errors: ``True`` for all of them (default), ``False`` for none, a
        sequence of output names, or a mapping from output name to the scale of the
        ``HalfNormal`` prior on the scatter. The scale that suits an output depends on
        how it was preprocessed, so pass a mapping when the default of 1.0 is a poor
        match.

    Examples
    --------
    A model for stellar labels and spectra sharing a 16-dimensional latent space, with
    a fitted per-pixel scatter on the flux only:

    >>> import pollux as plx
    >>> model = plx.Lux(
    ...     latent_size=16,
    ...     outputs={"label": 6, "flux": 1000},
    ...     intrinsic_scatter={"flux": 5.0},
    ... )
    >>> sorted(model.outputs)
    ['flux', 'label']
    >>> model.latent_size
    16

    The flux has a scatter to fit, the labels do not:

    >>> type(model.outputs["flux"].err_transform).__name__
    'ScatterTransform'
    >>> type(model.outputs["label"].err_transform).__name__
    'NoOpTransform'

    Making the flux quadratic in the latents:

    >>> model = plx.Lux(
    ...     latent_size=4, outputs={"flux": 100}, quadratic=["flux"]
    ... )
    >>> type(model.outputs["flux"].data_transform).__name__
    'QuadraticTransform'

    A Lux model is an :class:`~pollux.models.LVM`, so it is fitted and used the same way:

    >>> isinstance(model, plx.LVM)
    True
    """

    def __init__(
        self,
        latent_size: int,
        outputs: Mapping[str, int],
        quadratic: bool | Sequence[str] = False,
        intrinsic_scatter: bool | Sequence[str] | Mapping[str, float] = True,
    ) -> None:
        super().__init__(latent_size=latent_size)

        if not outputs:
            msg = (
                "Lux needs at least one output, given as a mapping from output name to "
                "output size, e.g. outputs={'flux': 1000}. To build up a model output "
                "by output instead, use pollux.LVM and its register_output() method."
            )
            raise ValueError(msg)

        is_quadratic = select_outputs(quadratic, outputs, "quadratic")
        scatter_scales = select_outputs(intrinsic_scatter, outputs, "intrinsic_scatter")

        for name, size in outputs.items():
            data_transform: AbstractSingleTransform = (
                QuadraticTransform(output_size=size)
                if name in is_quadratic
                else LinearTransform(output_size=size)
            )

            err_transform: ScatterTransform | None = None
            if name in scatter_scales:
                scale = scatter_scales[name]
                # No scale given means keep the transform's own default prior
                err_transform = (
                    ScatterTransform(output_size=size)
                    if scale is None
                    else ScatterTransform(
                        output_size=size, priors={"s": dist.HalfNormal(scale)}
                    )
                )

            self.register_output(name, data_transform, err_transform=err_transform)
