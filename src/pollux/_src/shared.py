import abc
import inspect
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

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


class AbstractModel(eqx.Module):
    outputs: dict[str, Output] = eqx.field(default_factory=dict, init=False)

    def register_output(
        self,
        name: str,
        size: int,
        transform: TransformT,
        transform_params: TransformParamsT,
    ) -> None:
        """Register a new output with its transform and parameter priors."""
        self.outputs[name] = Output(
            size=size, transform=transform, transform_params=transform_params
        )

    def _validate_data(
        self, data: dict[str, ArrayLike], errs: dict[str, ArrayLike]
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """Validate data and error dictionaries to make sure all values are JAX arrays.

        Parameters
        ----------
        data
            A dictionary of data values.
        errs
            A dictionary of error values.

        Returns
        -------
        tuple of dict
            A tuple of dictionaries (data, errs) where all values are JAX arrays.
        """
        # check that data and errs have the same keys
        if set(data.keys()) != set(errs.keys()):
            msg = "Data and errors must have the same keys"
            raise ModelValidationError(msg)

        # make sure values are jax arrays
        _data = {k: jnp.array(v) for k, v in data.items()}
        _errs = {k: jnp.array(v) for k, v in errs.items()}

        # make sure shapes of data and errs match
        for k in _data:
            if _data[k].shape != _errs[k].shape:
                msg = f"Data and errors for '{k}' must have the same shape"
                raise ModelValidationError(msg)
        return _data, _errs

    def unpack_numpyro_params(
        self, params: dict[str, jax.Array]
    ) -> dict[str, dict[str, jax.Array]]:
        """Unpack numpyro parameters into a nested structure.

        numpyro parameters use names like "output_name:param_name" to make the numpyro
        internal names unique. However, this method unpacks these into a nested
        dictionary keyed on [output_name][param_name].

        Parameters
        ----------
        params
            A dictionary of numpyro parameters. The keys should be in the format
            "output_name:param_name".

        Returns
        -------
        dict
            A nested dictionary of parameters, where the top level keys are the output
            names.
        """
        unpacked: dict[str, Any | dict[str, Any]] = {}
        handled_params = set()
        for name, output in self.outputs.items():
            unpacked[name] = {}
            for k in output.transform_params:
                numpyro_name = f"{name}:{k}"
                unpacked[name][k] = params[numpyro_name]
                handled_params.add(numpyro_name)

        for name in set(params.keys()) - handled_params:
            unpacked[name] = params[name]

        return unpacked

    def pack_numpyro_params(
        self, params: dict[str, dict[str, jax.Array]]
    ) -> dict[str, jax.Array]:
        """Pack parameters into a flat dictionary keyed on numpyro names.

        This method is the inverse of `unpack_numpyro_params`. It takes a nested
        dictionary of parameters and flattens it into a dictionary of numpyro
        parameters. For example, it takes a nested dictionary keyed like
        [output_name][param_name] and flattens it into a dictionary keyed like
        "output_name:param_name".

        Parameters
        ----------
        params
            A nested dictionary of parameters, where the top level keys are the output
            names.

        Returns
        -------
        dict
            A dictionary of numpyro parameters. The keys are in the format
            "output_name:param_name

        """
        packed: dict[str, jax.Array] = {}
        for name, output in self.outputs.items():
            for k in output.transform_params:
                numpyro_name = f"{name}:{k}"
                packed[numpyro_name] = params[name][k]

        return packed

    # Abstract methods to be implemented by the specific models:

    @abc.abstractmethod
    def setup_numpyro(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def default_numpyro_model(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def predict_outputs(self, *args: Any, **kwargs: Any) -> None:
        pass
