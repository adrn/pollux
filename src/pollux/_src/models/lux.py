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

    def unpack_pars(
        self, packed_pars: dict[str, Any], skip_missing: bool = False
    ) -> tuple[
        dict[str, Any] | tuple[dict[str, Any], ...],
        dict[str, Any] | tuple[dict[str, Any], ...],
    ]:
        """Unpack parameters for this output's data and error transforms.

        Parameters
        ----------
        packed_pars
            Dictionary of packed parameters with "err:" prefixed keys for error
            transform parameters.
        skip_missing
            If True, skip missing parameters instead of raising an error.

        Returns
        -------
        tuple
            A tuple of (data_pars, err_pars) where each element is either a
            dict (for single transforms) or a tuple of dicts (for transform sequences).
        """
        packed_data_pars: UnpackedParamsT = {}
        packed_err_pars: UnpackedParamsT = {}
        for name, value in packed_pars.items():
            if name.startswith("err:"):
                packed_err_pars[name[4:]] = value
            else:
                packed_data_pars[name] = value

        return self.data_transform.unpack_pars(
            packed_data_pars, skip_missing=skip_missing
        ), self.err_transform.unpack_pars(packed_err_pars, skip_missing=skip_missing)

    def pack_pars(
        self, unpacked_pars: dict[str, Any], skip_missing: bool = False
    ) -> PackedParamsT:
        """Pack data and error parameters for this output.

        Parameters
        ----------
        unpacked_pars
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
        data_pars = unpacked_pars.get("data", {})
        packed_data = self.data_transform.pack_pars(
            data_pars, skip_missing=skip_missing
        )
        packed.update(packed_data)

        # Pack error transform parameters with "err:" prefix
        err_pars = unpacked_pars.get("err", {})
        packed_err = self.err_transform.pack_pars(err_pars, skip_missing=skip_missing)
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
        data_pars: dict[str, Any],
        names: list[str] | str | None = None,
    ) -> BatchedOutputT | dict[str, BatchedOutputT]:
        """Predict output values for given latent vectors and parameters.

        Parameters
        ----------
        latents
            The latent vectors that transform into the outputs.
        data_pars
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
            if isinstance(data_pars[name], dict):
                results[name] = self.outputs[name].data_transform.apply(
                    latents, **data_pars[name]
                )
            else:
                results[name] = self.outputs[name].data_transform.apply(
                    latents, *data_pars[name]
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
        data_pars: dict[str, dict[str, jax.Array]] = {}
        err_pars: dict[str, dict[str, jax.Array]] = {}
        for output_name in output_names:
            # Priors for latent -> data transformation:
            data_priors[output_name] = self.outputs[
                output_name
            ].data_transform.get_expanded_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            data_pars[output_name] = {}
            for param_name, prior in data_priors[output_name].items():
                # Use new naming scheme: "output_name:param_name"
                # For TransformSequence, param_name already includes "{index}:{param}"
                numpyro_name = f"{output_name}:{param_name}"
                data_pars[output_name][param_name] = numpyro.sample(numpyro_name, prior)

            # Priors and parameters for transformation of the errors:
            err_priors[output_name] = self.outputs[
                output_name
            ].err_transform.get_expanded_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            err_pars[output_name] = {}
            for param_name, prior in err_priors[output_name].items():
                err_pars[output_name][param_name] = numpyro.sample(
                    f"{output_name}:err:{param_name}", prior
                )

        outputs = self.predict_outputs(latents, data_pars, names=output_names)
        for output_name in output_names:
            pred = outputs[output_name]

            # TODO NOTE: failure mode where .err is None and the err_transform doesn't
            # add a modeled intrinsic scatter. Detect this and raise an error?
            # TODO: This interface could be made more general to support, e.g.,
            # covariance matrices
            err = self.outputs[output_name].err_transform.apply(
                data[output_name].err, **err_pars[output_name]
            )
            numpyro.sample(
                f"obs:{output_name}",
                dist.Normal(pred, err),
                obs=data[output_name].data,
            )

        for output_name in output_names:
            data_pars[output_name].update(err_pars.get(output_name, {}))

        return data_pars

    def default_numpyro_model(
        self,
        data: PolluxData,
        latents_prior: dist.Distribution | None | bool = None,
        fixed_pars: PackedParamsT | None = None,
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
        latents_prior
            The prior distribution for the latent vectors. If not specified, use a unit
            Gaussian. If False, use an improper uniform prior.
        fixed_pars
            A dictionary of fixed parameters to condition on. If None, all parameters
            will be sampled.
        names
            A list of output names to include in the model. If None, include all outputs.
        custom_model
            Optional callable that takes latents, pars, and data and adds custom
            modeling components.
        """
        n_data = len(data)

        if latents_prior is None:
            _latents_prior = dist.Normal()

        elif latents_prior is False:
            _latents_prior = dist.ImproperUniform(
                dist.constraints.real,
                (),
                event_shape=(),
                sample_shape=(n_data,),
            )

        elif not isinstance(latents_prior, dist.Distribution):
            msg = "latents_prior must be a numpyro distribution instance"
            raise TypeError(msg)

        else:
            _latents_prior = latents_prior

        if _latents_prior.batch_shape != (self.latent_size,):
            _latents_prior = _latents_prior.expand((self.latent_size,))

        # Use condition handler to fix parameters if specified
        with numpyro.handlers.condition(data=fixed_pars or {}):
            latents = numpyro.sample(
                "latents",
                _latents_prior,
                sample_shape=(n_data,),
            )
            pars = self.setup_numpyro(latents, data, names=names)

        # Call the custom model function if provided
        if custom_model is not None:
            custom_model(latents, pars, data)

    def optimize(
        self,
        data: PolluxData,
        num_steps: int,
        rng_key: jax.Array,
        optimizer: OptimizerT | None = None,
        latents_prior: dist.Distribution | None | bool = None,
        custom_model: Callable[[BatchedLatentsT, dict[str, Any], PolluxData], None]
        | None = None,
        fixed_pars: UnpackedParamsT | None = None,
        names: list[str] | None = None,
        svi_run_kwargs: dict[str, Any] | None = None,
    ) -> tuple[UnpackedParamsT, Any]:
        """Optimize the model parameters.

        Parameters
        ----------

        """

        # Default to using Adam optimizer:
        optimizer = optimizer or numpyro.optim.Adam()

        partial_pars: dict[str, Any] = {}
        if fixed_pars is not None:
            packed_fixed_pars = self.pack_numpyro_pars(fixed_pars)
            partial_pars["fixed_pars"] = packed_fixed_pars

        if names is not None:
            partial_pars["names"] = names

        partial_pars["latents_prior"] = latents_prior
        partial_pars["custom_model"] = custom_model

        model: Any
        if partial_pars:
            model = partial(self.default_numpyro_model, **partial_pars)
        else:
            model = self.default_numpyro_model

        # The RNG key shouldn't have a massive impact here, since it it only used
        # internally by stochastic optimizers:
        svi_key, sample_key = jax.random.split(rng_key, 2)

        svi_run_kwargs = svi_run_kwargs or {}

        guide = AutoDelta(model)
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_results = svi.run(svi_key, num_steps, data, **svi_run_kwargs)
        packed_MAP_pars = guide.sample_posterior(sample_key, svi_results.pars)

        unpacked_pars = self.unpack_numpyro_pars(
            packed_MAP_pars,
            skip_missing=bool(fixed_pars is not None or names is not None),
        )
        # TODO: should the pars get their own object?
        return unpacked_pars, svi_results

    def unpack_numpyro_pars(
        self, pars: PackedParamsT, skip_missing: bool = False
    ) -> dict[str, Any]:
        """Unpack numpyro parameters into separate data and error parameter structures.

        numpyro parameters use names like "output_name:param_name" to make the numpyro
        internal names unique. This method unpacks these into two nested dictionaries:
        one for data transform parameters and one for error transform parameters.

        For TransformSequence outputs, data parameters are further unpacked from the
        flattened "{index}:{param}" format into a tuple of parameter dictionaries.

        Parameters
        ----------
        pars
            A dictionary of numpyro parameters. The keys should be in the format
            "output_name:param_name" or "output_name:err:param_name".
        skip_missing
            If True, skip parameters that are missing from the pars dict.

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
                "flux": {"data": {...} or (...), "err": {}},  # err empty if no error pars
                "label": {"data": {...}, "err": {...}},
                "latents": array
            }
        """
        unpacked_pars: dict[str, Any] = {}

        pars_by_output: dict[str, dict[str, Any]] = defaultdict(dict)
        for name, value in pars.items():
            if ":" in name:  # name associated with an output, like "flux:p1"
                output_name, *therest = name.split(":")

                if output_name not in self.outputs:
                    msg = (
                        f"Invalid output name {output_name} - expected one of: "
                        f"{list(self.outputs.keys())}"
                    )
                    raise ValueError(msg)

                pars_by_output[output_name][":".join(therest)] = value

            else:  # names not associated with outputs, like "latents", get passed thru
                unpacked_pars[name] = value

        for output, _pars in pars_by_output.items():
            data_pars, err_pars = self.outputs[output].unpack_pars(
                _pars, skip_missing=skip_missing
            )
            unpacked_pars[output] = {"data": data_pars, "err": err_pars}

        return unpacked_pars

    def pack_numpyro_pars(
        self,
        pars: dict[str, Any],  # TODO: update Any to real types
        skip_missing: bool = False,
    ) -> PackedParamsT:
        """Pack parameters into a flat dictionary keyed on numpyro names.

        This method is the inverse of `unpack_numpyro_pars`. It takes a nested
        dictionary of parameters and flattens it into a dictionary keyed on numpyro
        parameter names.

        Parameters
        ----------
        pars
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
            if output_name not in pars and not skip_missing:
                msg = f"Missing parameters for output {output_name}"
                raise ValueError(msg)

            output_pars = pars.get(output_name, {})
            tmp = output.pack_pars(
                {
                    "data": output_pars.get("data", {}),
                    "err": output_pars.get("err", {}),
                },
                skip_missing=skip_missing,
            )
            # Add output name prefix to all parameter keys
            for key, value in tmp.items():
                packed[f"{output_name}:{key}"] = value

        # Handle non-output parameters (like latents)
        for name in pars:
            if name not in self.outputs:
                packed[name] = pars[name]

        return packed
