import inspect
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import numpyro
import numpyro.distributions as dist

from ..data.data import PolluxData
from ..exceptions import ModelValidationError
from ..typing import TransformParamsT, TransformT


@dataclass
class ModelOutput:
    size: int
    transform: TransformT
    transform_params: TransformParamsT

    def __post_init__(self) -> None:
        # Validate transform parameters match signature
        sig = inspect.signature(self.transform)
        param_names = list(sig.parameters.keys())

        if len(param_names) < 1:
            msg = "transform must accept at least one argument (feature vector)"
            raise ModelValidationError(msg)

        # Skip first parameter (feature vector)
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

        # now we make sure to vmap over stars:
        self.transform = jax.vmap(
            self.transform, in_axes=(0, *tuple([None] * len(expected_params)))
        )


class PolluxModel(eqx.Module):
    """Blah

    Parameters
    ----------
    feature_size : int
        The size of the input feature vector.
    """

    feature_size: int
    outputs: dict[str, ModelOutput] = eqx.field(default_factory=dict, init=False)

    def register_output(
        self,
        name: str,
        size: int,
        transform: TransformT,
        transform_params: TransformParamsT,
    ) -> None:
        """Register a new output with its transform and parameter priors."""
        self.outputs[name] = ModelOutput(
            size=size, transform=transform, transform_params=transform_params
        )

    def predict_outputs(
        self,
        features: jax.Array,
        params: dict[str, Any],
        which: list[str] | str | None = None,
    ) -> jax.Array | dict[str, jax.Array]:
        """Predict outputs for given feature vectors and parameters.

        Parameters
        ----------
        features
            The feature vectors that transform into the outputs. In the case of the
            Paton, these are the (unknown) latent vectors. In the case of the Cannon,
            these are the observed features for the training set (combinations of
            stellar labels).
        params
            A dictionary of parameters for each output in the model.
        which
            A list of output names to predict. If None, predict all outputs.

        Returns
        -------
        dict
            A dictionary of predicted outputs, where the keys are the output names.
        """

        if which is None:
            which = list(self.outputs.keys())
        elif isinstance(which, str):
            which = [which]

        results = {}
        for name in which:
            output = self.outputs[name]

            # Note: needs to be a list or iterable because of how vmap works
            output_params = [params[name][k] for k in output.transform_params]
            results[name] = output.transform(features, *output_params)

        return results

    def setup_numpyro(
        self,
        features: jax.Array,
        data: PolluxData,
        which: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sample parameters and set up basic numpyro model.

        Parameters
        ----------
        features
            The feature vectors that transform into the outputs. In the case of the
            Paton, these are the (unknown) latent vectors. In the case of the Cannon,
            these are the observed features for the training set (combinations of
            stellar labels).
        data
            A dictionary of observed data.
        errs
            A dictionary of errors for the observed data.
        which
            A list of output names to include in the model. If None, use all outputs.

        Returns
        -------
        dict
            A dictionary of sampled parameters for each output.
        """
        which = which or list(self.outputs.keys())

        params: dict[str, dict[str, Any]] = {}
        for name in which:
            output = self.outputs[name]
            params[name] = {}
            for param_name, prior in output.transform_params.items():
                params[name][param_name] = numpyro.sample(f"{name}:{param_name}", prior)

        outputs = self.predict_outputs(features, params, which=which)
        for name in which:
            pred = outputs[name]
            numpyro.sample(
                f"obs:{name}",
                dist.Normal(pred, data[name].data_processed),
                obs=data[name].err_processed,
            )

        return params

    def default_numpyro_model(
        self,
        data: PolluxData,
        feature_prior: dist.Distribution | None | bool = None,
        fixed_params: dict[str, jax.Array] | None = None,
    ) -> None:
        """Create the default numpyro model for ...

        The default model uses the specified latent vector prior and assumes that the
        data are Gaussian distributed away from the true (predicted) values given the
        specified errors.

        Parameters
        ----------
        data
            A dictionary of observed data.
        errs
            A dictionary of errors for the observed data.
        feature_prior
            TODO
            The prior distribution for the latent vectors. If None, use a unit Gaussian.
        fixed_params
            A dictionary of fixed parameters to condition on. If None, all parameters
            will be sampled.

        """
        n_data = len(data)

        if feature_prior is None:
            _feature_prior = dist.Normal()

        elif feature_prior is False:
            _feature_prior = dist.ImproperUniform(
                dist.constraints.real,
                (),
                # event_shape=(self.latent_size,),  # TODO: or batch size?
                event_shape=(),
                sample_shape=(n_data,),
            )

        elif not isinstance(feature_prior, dist.Distribution):
            msg = "feature_prior must be a numpyro distribution instance"
            raise TypeError(msg)

        else:
            _feature_prior = feature_prior

        if _feature_prior.batch_shape != (self.feature_size,):
            _feature_prior = _feature_prior.expand((self.feature_size,))

        # Use condition handler to fix parameters if specified
        with numpyro.handlers.condition(data=fixed_params or {}):
            features = numpyro.sample(
                "features",
                _feature_prior,
                sample_shape=(n_data,),
            )
            self.setup_numpyro(features, data)

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
