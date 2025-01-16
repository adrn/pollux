"""The Paton model and associated functionality.

TODO:
- Docstrings and examples
- Is there a general way of registering the name of latent_z (paton) and features
  (cannon) so we can share the predict_outputs and setup_numpyro implementations?
"""

# import numpyro.distributions as dist

# from ..typing import TransformParamsT, TransformT


class Paton:
    """The Paton is a generative model for both stellar spectra and labels.

    In the Paton model, the features are latent (unknown) vectors of specified size that
    transform into both labels and fluxes. In other words, the features are a shared
    embedding for the labels and spectral flux data.

    Examples
    --------

    """

    # TODO: maybe this should be its own class that maintains an instance of PolluxModel

    # @classmethod
    # def default(
    #     cls,
    #     latent_size: int,
    #     data: PatonData,
    #     flux_transform: TransformT | None = None,
    #     flux_transform_params: TransformParamsT | None = None,
    #     label_transform: TransformT | None = None,
    #     label_transform_params: TransformParamsT | None = None,
    # ) -> "Paton":
    #     obj = cls(feature_size=latent_size)

    #     # The default transforms are linear with no offset:
    #     _flux_transform = flux_transform or (lambda z, A: A @ z)
    #     _flux_transform_params = flux_transform_params or {
    #         "A": dist.Normal(0, 1).expand((data.flux.shape[1], obj.feature_size)),
    #     }
    #     _label_transform = label_transform or (lambda z, A: A @ z)
    #     _label_transform_params = label_transform_params or {
    #         "A": dist.Normal(0, 1).expand((data.label.shape[1], obj.feature_size)),
    #     }

    #     obj.register_output(
    #         "labels",
    #         size=data.label.shape[1],
    #         transform=_label_transform,  # type: ignore[arg-type]
    #         transform_params=_label_transform_params,
    #     )
    #     obj.register_output(
    #         "flux",
    #         size=data.flux.shape[1],
    #         transform=_flux_transform,  # type: ignore[arg-type]
    #         transform_params=_flux_transform_params,
    #     )

    #     return obj

    # # TODO: functionality to train the model
    # # TODO: functionality to sample from the prior and initialize parameters
    # # TODO: functionality to predict flux from labels (through the latents) and vice versa
