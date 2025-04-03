from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from ..data import PolluxData
from ..typing import (
    BatchedLatentsT,
    BatchedOutputT,
    OptimizerT,
    PackedParamsT,
    UnpackedParamsT,
)
from .transforms import AbstractTransform


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
    outputs: dict[str, AbstractTransform] = eqx.field(default_factory=dict, init=False)

    def register_output(self, name: str, transform: AbstractTransform) -> None:
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
        if name in self.outputs:
            msg = f"Output with name {name} already exists"
            raise ValueError(msg)
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
            if isinstance(params[name], dict):
                results[name] = self.outputs[name].apply(latents, **params[name])
            else:
                results[name] = self.outputs[name].apply(latents, *params[name])

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

        priors: dict[str, dict[str, Any]] = {}
        params: dict[str, dict[str, jax.Array]] = {}
        for name in names:
            priors[name] = self.outputs[name].get_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            params[name] = {}
            for param_name, prior in priors[name].items():
                params[name][param_name] = numpyro.sample(f"{name}:{param_name}", prior)

        outputs = self.predict_outputs(latents, params, names=names)
        for name in names:
            pred = outputs[name]
            numpyro.sample(
                f"obs:{name}",
                dist.Normal(
                    pred,
                    jnp.sqrt(data[name].err ** 2 + params[name].get("s", 0.0) ** 2),  # type: ignore[operator]
                ),  # NOTE: s is the intrinsic scatter or excess variance
                obs=data[name].data,
            )

        return params

    def default_numpyro_model(
        self,
        data: PolluxData,
        latent_prior: dist.Distribution | None | bool = None,
        fixed_params: PackedParamsT | None = None,
        names: list[str] | None = None,
        custom_model: Callable[[BatchedLatentsT, dict[str, Any], PolluxData], None]
        | None = None,
    ) -> None:
        """Create the default numpyro model for this Lux model.

        The default model uses the specified latent vector prior and assumes that the
        data are Gaussian distributed away from the true (predicted) values given the
        specified errors.

        Parameters
        ----------
        data
            A dictionary of observed data.
        latent_prior
            The prior distribution for the latent vectors. If not specified, use a unit
            Gaussian. If False, use an improper uniform prior.
        fixed_params
            A dictionary of fixed parameters to condition on. If None, all parameters
            will be sampled.
        names
            A list of output names to include in the model. If None, include all outputs.
        custom_model
            Optional callable that takes latents, params, and data and adds custom
            modeling components.
        """
        n_data = len(data)

        if latent_prior is None:
            _latent_prior = dist.Normal()

        elif latent_prior is False:
            _latent_prior = dist.ImproperUniform(
                dist.constraints.real,
                (),
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
            params = self.setup_numpyro(latents, data, names=names)

        # Call the custom model function if provided
        if custom_model is not None:
            custom_model(latents, params, data)

    def optimize(
        self,
        data: PolluxData,
        num_steps: int,
        rng_key: jax.Array,
        optimizer: OptimizerT | None = None,
        latent_prior: dist.Distribution | None | bool = None,
        custom_model: Callable[[BatchedLatentsT, dict[str, Any], PolluxData], None]
        | None = None,
        fixed_params: UnpackedParamsT | None = None,
        names: list[str] | None = None,
        svi_run_kwargs: dict[str, Any] | None = None,
    ) -> tuple[UnpackedParamsT, Any]:
        """Optimize the model parameters.

        Parameters
        ----------

        """

        # Default to using Adam optimizer:
        optimizer = optimizer or numpyro.optim.Adam()

        partial_params: dict[str, Any] = {}
        if fixed_params is not None:
            packed_fixed_params = self.pack_numpyro_params(fixed_params)
            partial_params["fixed_params"] = packed_fixed_params

        if names is not None:
            partial_params["names"] = names

        partial_params["latent_prior"] = latent_prior
        partial_params["custom_model"] = custom_model

        model: Any
        if partial_params:
            model = partial(self.default_numpyro_model, **partial_params)
        else:
            model = self.default_numpyro_model

        # The RNG key shouldn't have a massive impact here, since it it only used
        # internally by stochastic optimizers:
        svi_key, sample_key = jax.random.split(rng_key, 2)

        svi_run_kwargs = svi_run_kwargs or {}

        guide = AutoDelta(model)
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_results = svi.run(svi_key, num_steps, data, **svi_run_kwargs)
        unpacked_MAP_params = guide.sample_posterior(sample_key, svi_results.params)
        return self.unpack_numpyro_params(
            unpacked_MAP_params, skip_missing=True
        ), svi_results

    def unpack_numpyro_params(
        self, params: PackedParamsT, skip_missing: bool = False
    ) -> UnpackedParamsT:
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
                if numpyro_name not in params and skip_missing:
                    continue
                unpacked[name][k] = params[numpyro_name]
                handled_params.add(numpyro_name)

        for name in set(params.keys()) - handled_params:
            unpacked[name] = params[name]

        return unpacked

    def pack_numpyro_params(self, params: UnpackedParamsT) -> PackedParamsT:
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
