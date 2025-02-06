from typing import Any

import equinox as eqx
import jax
import numpyro
import numpyro.distributions as dist

from ..data import PolluxData
from ..typing import BatchedLatentsT, BatchedOutputT
from .transforms import AbstractOutputTransform


class LuxModel(eqx.Module):
    """A latent variable model with multiple outputs.

    A Pollux model is a generative, latent variable model for output data. This is a
    general framework for constructing multi-output or multi-task models in which the
    output data is generated as a transformation away from some embedded vector
    representation of each object. While this class and model structure can be used in a
    broad range of applications, this package and implementation was written with
    applications to stellar spectroscopic data in mind.

    Parameters
    ----------
    latent_size : int
        The size of the latent vector representation for each object (i.e. the embedded
        dimensionality).
    """

    latent_size: int
    outputs: dict[str, AbstractOutputTransform] = eqx.field(
        default_factory=dict, init=False
    )

    def register_output(self, name: str, transform: AbstractOutputTransform) -> None:
        """Register a new output of the model given a specified transform.

        Parameters
        ----------
        name
            The name of the output. If you intend to use this model with numpyro and
            specified data, this name should correspond to the name of data passed in
            via a `pollux.data.PolluxData` object.
        transform
            A specification of the transformation function that takes a latent vector
            representation in and predicts the output values.
        """
        self.outputs[name] = transform

    def predict_outputs(
        self,
        latents: BatchedLatentsT,
        params: dict[str, Any],
        names: list[str] | str | None = None,
    ) -> BatchedOutputT | dict[str, BatchedOutputT]:
        """Predict output values for given latent vectors and parameters.

        Parameters
        ----------
        latents
            The latent vectors that transform into the outputs.
        params
            A dictionary of parameters for each output transformation in the model.
        names
            A single string or a list of output names to predict. If None, predict all
            outputs (default).

        Returns
        -------
        dict
            A dictionary of predicted output values, where the keys are the output
            names.
        """

        if latents.shape[-1] != self.latent_size:
            msg = (
                f"Latent vectors have size {latents.shape[-1]} along their final axis, "
                f"but expected them to have size {self.latent_size} "
            )
            raise ValueError(msg)

        if names is None:
            names = list(self.outputs.keys())
        elif isinstance(names, str):
            names = [names]

        results = {}
        for name in names:
            results[name] = self.outputs[name].apply(latents, **params[name])

        return results

    def setup_numpyro(
        self,
        latents: BatchedLatentsT,
        data: PolluxData,
        names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sample parameters and set up basic numpyro model.

        Parameters
        ----------
        latents
            The latent vectors that transform into the outputs. In the case of the
            Paton, these are the (unknown) latent vectors. In the case of the Cannon,
            these are the observed latents for the training set (combinations of
            stellar labels).
        data
            A dictionary-like object of observed data for each output. The keys should
            correspond to the output names.
        names
            A single string or a list of output names to set up. If None, set up all
            outputs (default).

        Returns
        -------
        dict
            A dictionary of sampled parameters for each output.
        """
        names = names or list(self.outputs.keys())

        params: dict[str, dict[str, Any]] = {}
        for name in names:
            priors = self.outputs[name].get_priors(self.latent_size)
            params[name] = {}
            for param_name, prior in priors.items():
                params[name][param_name] = numpyro.sample(f"{name}:{param_name}", prior)

        outputs = self.predict_outputs(latents, params, names=names)
        for name in names:
            pred = outputs[name]
            numpyro.sample(
                f"obs:{name}",
                dist.Normal(pred, data._err_processed[name]),
                obs=data._data_processed[name],
            )

        return params

    def default_numpyro_model(
        self,
        data: PolluxData,
        latent_prior: dist.Distribution | None | bool = None,
        fixed_params: dict[str, jax.Array] | None = None,
    ) -> None:
        """Create the default numpyro model for this Lux model.

        The default model uses the specified latent vector prior and assumes that the
        data are Gaussian distributed away from the true (predicted) values given the
        specified errors.

        Parameters
        ----------
        data
            A dictionary of observed data.
        errs
            A dictionary of errors for the observed data.
        latent_prior
            The prior distribution for the latent vectors. If not specified, use a unit
            Gaussian. If False, use an improper uniform prior.
        fixed_params
            A dictionary of fixed parameters to condition on. If None, all parameters
            will be sampled.

        """
        n_data = len(data)

        if latent_prior is None:
            _latent_prior = dist.Normal()

        elif latent_prior is False:
            _latent_prior = dist.ImproperUniform(
                dist.constraints.real,
                (),
                # event_shape=(self.latent_size,),  # TODO: or batch size?
                event_shape=(),
                sample_shape=(n_data,),
            )

        elif not isinstance(latent_prior, dist.Distribution):
            msg = "latent_prior must be a numpyro distribution instance"
            raise TypeError(msg)

        else:
            _latent_prior = latent_prior

        if _latent_prior.batch_shape != (self.latent_size,):
            _latent_prior = _latent_prior.expand((self.latent_size,))

        # Use condition handler to fix parameters if specified
        with numpyro.handlers.condition(data=fixed_params or {}):
            latents = numpyro.sample(
                "latents",
                _latent_prior,
                sample_shape=(n_data,),
            )
            self.setup_numpyro(latents, data)

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
            for k in output.param_priors:
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
            for k in output.param_priors:
                numpyro_name = f"{name}:{k}"
                packed[numpyro_name] = params[name][k]

        return packed
