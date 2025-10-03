from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import equinox as eqx
import jax
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
from .transforms import AbstractSingleTransform, NoOpTransform, TransformSequence


class LuxOutput(eqx.Module):
    data_transform: AbstractSingleTransform | TransformSequence
    err_transform: AbstractSingleTransform | TransformSequence

    def unpack_params(
        self, packed_params: dict[str, Any], skip_missing: bool = False
    ) -> tuple[
        dict[str, Any] | tuple[dict[str, Any], ...],
        dict[str, Any] | tuple[dict[str, Any], ...],
    ]:
        """Unpack parameters for this output's data and error transforms.

        Parameters
        ----------
        packed_params
            Dictionary of packed parameters with "err:" prefixed keys for error
            transform parameters.
        skip_missing
            If True, skip missing parameters instead of raising an error.

        Returns
        -------
        tuple
            A tuple of (data_params, err_params) where each element is either a
            dict (for single transforms) or a tuple of dicts (for transform sequences).
        """
        packed_data_params: UnpackedParamsT = {}
        packed_err_params: UnpackedParamsT = {}
        for name, value in packed_params.items():
            if name.startswith("err:"):
                packed_err_params[name[4:]] = value
            else:
                packed_data_params[name] = value

        return self.data_transform.unpack_params(
            packed_data_params, skip_missing=skip_missing
        ), self.err_transform.unpack_params(
            packed_err_params, skip_missing=skip_missing
        )

    def pack_params(
        self, unpacked_params: dict[str, Any], skip_missing: bool = False
    ) -> PackedParamsT:
        """Pack data and error parameters for this output.

        Parameters
        ----------
        unpacked_params
            Dictionary with "data" and "err" keys containing the unpacked parameters
            for the data and error transforms respectively.
        skip_missing
            If True, skip missing parameters instead of raising an error.

        Returns
        -------
        dict
            Flat dictionary with parameter names that include any necessary prefixes
            (e.g., "err:" for error parameters, "0:" for TransformSequence indices).
        """
        packed: dict[str, jax.Array] = {}

        # Pack data transform parameters
        data_params = unpacked_params.get("data", {})
        packed_data = self.data_transform.pack_params(
            data_params, skip_missing=skip_missing
        )
        packed.update(packed_data)

        # Pack error transform parameters with "err:" prefix
        err_params = unpacked_params.get("err", {})
        packed_err = self.err_transform.pack_params(
            err_params, skip_missing=skip_missing
        )
        for key, value in packed_err.items():
            packed[f"err:{key}"] = value

        return packed


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
    outputs: dict[str, LuxOutput] = eqx.field(default_factory=dict, init=False)

    def register_output(
        self,
        name: str,
        data_transform: AbstractSingleTransform | TransformSequence,
        err_transform: AbstractSingleTransform | TransformSequence | None = None,
    ) -> None:
        """Register a new output of the model given a specified transform.

        Parameters
        ----------
        name
            The name of the output. If you intend to use this model with numpyro and
            specified data, this name should correspond to the name of data passed in
            via a `pollux.data.PolluxData` object.
        data_transform
            A specification of the transformation function that takes a latent vector
            representation in and predicts the output values.
        """
        if name in self.outputs:
            msg = f"Output with name {name} already exists"
            raise ValueError(msg)
        if err_transform is None:
            err_transform = NoOpTransform()
        self.outputs[name] = LuxOutput(data_transform, err_transform)

    def predict_outputs(
        self,
        latents: BatchedLatentsT,
        data_params: dict[str, Any],
        names: list[str] | str | None = None,
    ) -> BatchedOutputT | dict[str, BatchedOutputT]:
        """Predict output values for given latent vectors and parameters.

        Parameters
        ----------
        latents
            The latent vectors that transform into the outputs.
        data_params
            A dictionary of parameters for the data component of each output
            transformation in the model. For TransformSequence outputs, expects either:
            - A list of parameter dictionaries
            - A flat dictionary with "{index}:{param}" keys
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
            if isinstance(data_params[name], dict):
                results[name] = self.outputs[name].data_transform.apply(
                    latents, **data_params[name]
                )
            else:
                results[name] = self.outputs[name].data_transform.apply(
                    latents, *data_params[name]
                )

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
        output_names = names or list(self.outputs.keys())

        data_priors: dict[str, Mapping[str, Any]] = {}
        err_priors: dict[str, Mapping[str, Any]] = {}
        data_params: dict[str, dict[str, jax.Array]] = {}
        err_params: dict[str, dict[str, jax.Array]] = {}
        for output_name in output_names:
            # Priors for latent -> data transformation:
            data_priors[output_name] = self.outputs[
                output_name
            ].data_transform.get_expanded_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            data_params[output_name] = {}
            for param_name, prior in data_priors[output_name].items():
                # Use new naming scheme: "output_name:param_name"
                # For TransformSequence, param_name already includes "{index}:{param}"
                numpyro_name = f"{output_name}:{param_name}"
                data_params[output_name][param_name] = numpyro.sample(
                    numpyro_name, prior
                )

            # Priors and parameters for transformation of the errors:
            err_priors[output_name] = self.outputs[
                output_name
            ].err_transform.get_expanded_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            err_params[output_name] = {}
            for param_name, prior in err_priors[output_name].items():
                err_params[output_name][param_name] = numpyro.sample(
                    f"{output_name}:err:{param_name}", prior
                )

        outputs = self.predict_outputs(latents, data_params, names=output_names)
        for output_name in output_names:
            pred = outputs[output_name]

            # TODO NOTE: failure mode where .err is None and the err_transform doesn't
            # add a modeled intrinsic scatter. Detect this and raise an error?
            # TODO: This interface could be made more general to support, e.g.,
            # covariance matrices
            err = self.outputs[output_name].err_transform.apply(
                data[output_name].err, **err_params[output_name]
            )
            numpyro.sample(
                f"obs:{output_name}",
                dist.Normal(pred, err),
                obs=data[output_name].data,
            )

        for output_name in output_names:
            data_params[output_name].update(err_params.get(output_name, {}))

        return data_params

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
        packed_MAP_params = guide.sample_posterior(sample_key, svi_results.params)

        unpacked_params = self.unpack_numpyro_params(
            packed_MAP_params,
            skip_missing=bool(fixed_params is not None or names is not None),
        )
        # TODO: should the params get their own object?
        return unpacked_params, svi_results

    def unpack_numpyro_params(
        self, params: PackedParamsT, skip_missing: bool = False
    ) -> dict[str, Any]:
        """Unpack numpyro parameters into separate data and error parameter structures.

        numpyro parameters use names like "output_name:param_name" to make the numpyro
        internal names unique. This method unpacks these into two nested dictionaries:
        one for data transform parameters and one for error transform parameters.

        For TransformSequence outputs, data parameters are further unpacked from the
        flattened "{index}:{param}" format into a tuple of parameter dictionaries.

        Parameters
        ----------
        params
            A dictionary of numpyro parameters. The keys should be in the format
            "output_name:param_name" or "output_name:err:param_name".
        skip_missing
            If True, skip parameters that are missing from the params dict.

        Returns
        -------
        dict
            A nested dictionary with keys as output names. Each output name is a key with
            a dict value containing "data" and "err" keys:
            - For single transforms, "data" values are parameter dictionaries
            - For TransformSequence, "data" values are tuples of parameter dictionaries
            - "err" values follow the same structure as "data" for the error transforms
            - "err" will be an empty dict {} if there are no error parameters
            - Non-output parameters (like "latents") are passed through at the top level

            Example structure:
            {
                "flux": {"data": {...} or (...), "err": {}},  # err empty if no error params
                "label": {"data": {...}, "err": {...}},
                "latents": array
            }
        """
        unpacked_params: dict[str, Any] = {}

        params_by_output: dict[str, dict[str, Any]] = defaultdict(dict)
        for name, value in params.items():
            if ":" in name:  # name associated with an output, like "flux:p1"
                output_name, *therest = name.split(":")

                if output_name not in self.outputs:
                    msg = (
                        f"Invalid output name {output_name} - expected one of: "
                        f"{list(self.outputs.keys())}"
                    )
                    raise ValueError(msg)

                params_by_output[output_name][":".join(therest)] = value

            else:  # names not associated with outputs, like "latents", get passed thru
                unpacked_params[name] = value

        for output, pars in params_by_output.items():
            data_params, err_params = self.outputs[output].unpack_params(
                pars, skip_missing=skip_missing
            )
            unpacked_params[output] = {"data": data_params, "err": err_params}

        return unpacked_params

    def pack_numpyro_params(
        self,
        params: dict[str, Any],  # TODO: update Any to real types
        skip_missing: bool = False,
    ) -> PackedParamsT:
        """Pack parameters into a flat dictionary keyed on numpyro names.

        This method is the inverse of `unpack_numpyro_params`. It takes a nested
        dictionary of parameters and flattens it into a dictionary keyed on numpyro
        parameter names.

        Parameters
        ----------
        params
            A nested dictionary with keys as output names. Each output name should
            be a key with a dict value containing "data" and optionally "err" keys.
            The "err" key can be omitted if there are no error parameters for that output.
            For TransformSequence outputs, "data" values should be tuples/lists of
            parameter dictionaries. Non-output parameters (like "latents") can exist at
            the top level.

            Example structure:
            {
                "flux": {"data": {...} or (...)},  # err key optional
                "label": {"data": {...}, "err": {...}},  # err key included
                "latents": array
            }

        Returns
        -------
        dict
            A dictionary of numpyro parameters. The keys are in the format
            "output_name:param_name" for data parameters and "output_name:err:param_name"
            for error parameters.
        """
        packed: dict[str, jax.Array] = {}

        for output_name, output in self.outputs.items():
            if output_name not in params and not skip_missing:
                msg = f"Missing parameters for output {output_name}"
                raise ValueError(msg)

            output_params = params.get(output_name, {})
            tmp = output.pack_params(
                {
                    "data": output_params.get("data", {}),
                    "err": output_params.get("err", {}),
                },
                skip_missing=skip_missing,
            )
            # Add output name prefix to all parameter keys
            for key, value in tmp.items():
                packed[f"{output_name}:{key}"] = value

        # Handle non-output parameters (like latents)
        for name in params:
            if name not in self.outputs:
                packed[name] = params[name]

        return packed
