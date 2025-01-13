import inspect
from dataclasses import dataclass

import jax
import numpyro.distributions as dist

from .exceptions import ModelValidationError
from .typing import TransformParamsT, TransformT


@dataclass
class Output:
    size: int
    transform: TransformT
    transform_params: TransformParamsT

    def __post_init__(self) -> None:
        # Validate transform parameters match signature
        sig = inspect.signature(self.transform)
        param_names = list(sig.parameters.keys())

        if len(param_names) < 1:
            msg = "transform must accept at least one argument (latent vector)"
            raise ModelValidationError(msg)

        # Skip first parameter (latent vector)
        expected_params = set(param_names[1:])
        provided_params = set(self.transform_params.keys())

        if expected_params != provided_params:
            missing = expected_params - provided_params
            extra = provided_params - expected_params
            msgs = []
            if missing:
                msgs.append(f"Missing parameters: {missing}")
            if extra:
                msgs.append(f"Unexpected parameters: {extra}")
            raise ModelValidationError(". ".join(msgs))

        # Validate all priors are proper distributions
        for name, prior in self.transform_params.items():
            if not isinstance(prior, dist.Distribution):
                msg = f"Prior '{name}' must be a numpyro Distribution instance"
                raise ModelValidationError(msg)

        # TODO: make sure transform_params is in the order of the transform signature
        # (or at least that the order of the keys matches the order of the signature)

        # now we make sure to vmap over latents:
        self.transform = jax.vmap(
            self.transform, in_axes=(0, *tuple([None] * len(expected_params)))
        )
