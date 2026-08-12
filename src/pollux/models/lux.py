import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoGuide

from ..data import PolluxData
from ..typing import (
    BatchedLatentsT,
    BatchedOutputT,
    OptimizerT,
    PackedParamsT,
    UnpackedParamsT,
)
from .iterative import optimize_iterative
from .transforms import AbstractSingleTransform, NoOpTransform, TransformSequence

if TYPE_CHECKING:
    from .iterative import IterativeOptimizationResult, ParameterBlock


class LuxOutput(eqx.Module):
    data_transform: AbstractSingleTransform | TransformSequence
    err_transform: AbstractSingleTransform | TransformSequence

    def unpack_pars(
        self, packed_pars: dict[str, Any], ignore_missing: bool = False
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
        ignore_missing
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
            packed_data_pars, ignore_missing=ignore_missing
        ), self.err_transform.unpack_pars(
            packed_err_pars, ignore_missing=ignore_missing
        )

    def pack_pars(
        self, unpacked_pars: dict[str, Any], ignore_missing: bool = False
    ) -> PackedParamsT:
        """Pack data and error parameters for this output.

        Parameters
        ----------
        unpacked_pars
            Dictionary with "data" and "err" keys containing the unpacked parameters
            for the data and error transforms respectively.
        ignore_missing
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
            data_pars, ignore_missing=ignore_missing
        )
        packed.update(packed_data)

        # Pack error transform parameters with "err:" prefix
        err_pars = unpacked_pars.get("err", {})
        packed_err = self.err_transform.pack_pars(
            err_pars, ignore_missing=ignore_missing
        )
        for key, value in packed_err.items():
            packed[f"err:{key}"] = value

        return packed


class Lux(eqx.Module):
    """A latent variable model with multiple outputs.

    Lux is a generative, latent variable model for output data. This is a general
    framework for constructing multi-output or multi-task models in which the output
    data is generated as a transformation away from some embedded vector representation
    of each object. While this class and model structure can be used in a broad range of
    applications, this package and implementation was written with applications to
    stellar spectroscopic data in mind.

    Parameters
    ----------
    latent_size : int
        The size of the latent vector representation for each object (i.e. the embedded
        dimensionality).

    Notes
    -----
    **Parameter Format**

    The :meth:`optimize` method returns parameters in a nested format::

        {
            "output_name": {
                "data": {"A": array, ...},  # Transform parameters
                "err": {"s": array, ...}    # Error transform parameters
            },
            "latents": array  # Per-object latent vectors
        }

    This same format should be used when passing parameters to :meth:`predict_outputs`.

    **Naming Restrictions**

    Output names and transform parameter names cannot contain colons (``':'``) as they
    are reserved for internal parameter naming in numpyro.
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
            via a `pollux.data.PolluxData` object. The name cannot contain colons (':')
            as they are reserved for internal parameter naming.
        data_transform
            A specification of the transformation function that takes a latent vector
            representation in and predicts the output values.
        """
        if ":" in name:
            msg = (
                f"Output name '{name}' contains ':' which is reserved for internal "
                "parameter naming. Please use a different name."
            )
            raise ValueError(msg)
        if name in self.outputs:
            msg = f"Output with name {name} already exists"
            raise ValueError(msg)
        if err_transform is None:
            err_transform = NoOpTransform()
        self.outputs[name] = LuxOutput(data_transform, err_transform)

    def predict_outputs(
        self,
        latents: BatchedLatentsT,
        pars: dict[str, Any],
        names: list[str] | str | None = None,
    ) -> BatchedOutputT | dict[str, BatchedOutputT]:
        """Predict output values for given latent vectors and parameters.

        Parameters
        ----------
        latents
            The latent vectors that transform into the outputs. Shape should be
            ``(n_objects, latent_size)``.
        pars
            A dictionary of parameters for each output transformation in the model.
            Should be in the nested format returned by :meth:`optimize`::

                {
                    "output_name": {
                        "data": {...} or [...],  # Transform parameters
                        "err": {...}             # Error transform parameters
                    },
                    "latents": array  # Optional, not used here
                }

            For single transforms, ``"data"`` is a dict: ``{"A": array, "b": array}``

            For :class:`TransformSequence`, ``"data"`` is a tuple of dicts:
            ``({"A": array}, {"b": array})``

            .. deprecated::
                Passing parameters in direct format (without the ``"data"``/``"err"``
                wrapper) is deprecated and will be removed in a future version.

        names
            A single string or a list of output names to predict. If ``None``, predict
            all outputs (default).

        Returns
        -------
        dict
            A dictionary of predicted output values, where the keys are the output
            names and values are arrays of shape ``(n_objects, output_size)``.
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

        # Extract data parameters, handling both nested and direct formats
        data_pars = {}
        direct_format = False
        for name in names:
            output_pars = pars[name]
            if not isinstance(output_pars, dict):
                msg = (
                    f"Expected dict for parameters of output '{name}', "
                    f"got {type(output_pars).__name__}"
                )
                raise TypeError(msg)

            if "data" in output_pars or "err" in output_pars:
                data_pars[name] = output_pars.get("data", {})
            else:
                direct_format = True
                data_pars[name] = output_pars

        if direct_format:
            warnings.warn(
                "Passing parameters in direct format (e.g., {'flux': {'A': ...}}) is "
                "deprecated. Please use the nested format returned by optimize(): "
                "{'flux': {'data': {'A': ...}, 'err': {...}}}. "
                "Direct format support will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

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

        data_pars: dict[str, dict[str, jax.Array]] = {}
        err_pars: dict[str, dict[str, jax.Array]] = {}
        for output_name in output_names:
            output = self.outputs[output_name]

            # Priors for latent -> data transformation. Use the naming scheme
            # "output_name:param_name"; for a TransformSequence, param_name
            # already includes its own "{index}:{param}" prefix.
            data_priors = output.data_transform.get_expanded_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            data_pars[output_name] = {
                param_name: numpyro.sample(f"{output_name}:{param_name}", prior)
                for param_name, prior in data_priors.items()
            }

            # Priors and parameters for transformation of the errors:
            err_priors = output.err_transform.get_expanded_priors(
                latent_size=self.latent_size, data_size=len(data)
            )
            err_pars[output_name] = {
                param_name: numpyro.sample(f"{output_name}:err:{param_name}", prior)
                for param_name, prior in err_priors.items()
            }

        # Wrap data_pars in nested format for predict_outputs
        nested_pars = {k: {"data": v} for k, v in data_pars.items()}
        outputs = self.predict_outputs(latents, nested_pars, names=output_names)
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
        guide: type[AutoGuide] | AutoGuide | None = None,
    ) -> tuple[UnpackedParamsT, Any]:
        """Optimize the model parameters using SVI.

        Parameters
        ----------
        data
            The observed data to optimize against.
        num_steps
            Number of SVI optimization steps.
        rng_key
            JAX random key for the optimization.
        optimizer
            Numpyro optimizer to use. Defaults to
            ``numpyro.optim.Adam(step_size=1e-3)``.
        latents_prior
            Prior distribution for the latent vectors. If ``None``, uses a unit
            Gaussian. If ``False``, uses an improper uniform prior.
        custom_model
            Optional callable for custom modeling components.
        fixed_pars
            Parameters to hold fixed during optimization.
        names
            Output names to include. If ``None``, includes all outputs.
        svi_run_kwargs
            Additional keyword arguments passed to ``SVI.run()``.
        guide
            The autoguide to use for variational inference. Can be:

            - ``None`` (default): uses ``AutoDelta`` for MAP estimation.
            - A guide class (e.g. ``AutoNormal``): will be instantiated with the
              model function.
            - A guide instance: used directly (must already be constructed with
              the model function).

        """

        # Default to using Adam optimizer. numpyro's Adam has no default step
        # size; 1e-3 matches the default used by optimize_iterative's SVI blocks.
        optimizer = optimizer or numpyro.optim.Adam(step_size=1e-3)

        # ignore_missing=True: fixed_pars typically holds only a subset of the
        # parameters, namely the ones to hold fixed during optimization
        model: Any = partial(
            self.default_numpyro_model,
            fixed_pars=(
                None
                if fixed_pars is None
                else self.pack_numpyro_pars(fixed_pars, ignore_missing=True)
            ),
            names=names,
            latents_prior=latents_prior,
            custom_model=custom_model,
        )

        # The RNG key shouldn't have a massive impact here, since it it only used
        # internally by stochastic optimizers:
        svi_key, sample_key = jax.random.split(rng_key, 2)

        svi_run_kwargs = svi_run_kwargs or {}

        if guide is None:
            _guide = AutoDelta(model)
        elif isinstance(guide, type) and issubclass(guide, AutoGuide):
            _guide = guide(model)
        elif isinstance(guide, AutoGuide):
            _guide = guide
        else:
            msg = (
                "guide must be None, an AutoGuide subclass, or an AutoGuide instance, "
                f"got {type(guide)}"
            )
            raise TypeError(msg)

        svi = SVI(model, _guide, optimizer, Trace_ELBO())
        svi_results = svi.run(svi_key, num_steps, data, **svi_run_kwargs)
        packed_MAP_pars = _guide.sample_posterior(sample_key, svi_results.params)

        unpacked_pars = self.unpack_numpyro_pars(
            packed_MAP_pars,
            ignore_missing=bool(fixed_pars is not None or names is not None),
        )
        # TODO: should the pars get their own object?
        return unpacked_pars, svi_results

    def optimize_iterative(
        self,
        data: PolluxData,
        blocks: "list[ParameterBlock] | list[str] | None" = None,
        fixed_pars: UnpackedParamsT | None = None,
        max_cycles: int = 10,
        tol: float = 1e-4,
        rng_key: jax.Array | None = None,
        initial_params: UnpackedParamsT | None = None,
        latents_prior: dist.Distribution | None = None,
        progress: bool = True,
    ) -> "IterativeOptimizationResult":
        """Optimize using iterative parameter block coordinate descent.

        For models with purely linear outputs, this exploits the linear structure
        for faster convergence: each sub-problem is solved exactly using weighted
        least squares. The default strategy alternates between optimizing the
        latents (with output parameters fixed) and optimizing each output's
        parameters (with the latents fixed).

        See :func:`pollux.models.optimize_iterative` for the full description of
        the parameters and of the returned result. Note that this method defaults
        to ``max_cycles=10``.
        """
        return optimize_iterative(
            model=self,
            data=data,
            blocks=blocks,
            fixed_pars=fixed_pars,
            max_cycles=max_cycles,
            tol=tol,
            rng_key=rng_key,
            initial_params=initial_params,
            latents_prior=latents_prior,
            progress=progress,
        )

    def unpack_numpyro_pars(
        self, pars: PackedParamsT, ignore_missing: bool = False
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
        ignore_missing
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
                _pars, ignore_missing=ignore_missing
            )
            unpacked_pars[output] = {"data": data_pars, "err": err_pars}

        return unpacked_pars

    def pack_numpyro_pars(
        self,
        pars: UnpackedParamsT,
        ignore_missing: bool = False,
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
            if output_name not in pars and not ignore_missing:
                msg = f"Missing parameters for output {output_name}"
                raise ValueError(msg)

            output_pars = dict(pars.get(output_name, {}))
            tmp = output.pack_pars(
                {
                    "data": output_pars.get("data", {}),
                    "err": output_pars.get("err", {}),
                },
                ignore_missing=ignore_missing,
            )
            # Add output name prefix to all parameter keys
            for key, value in tmp.items():
                packed[f"{output_name}:{key}"] = value

        # Handle non-output parameters (like latents)
        for name in pars:
            if name not in self.outputs:
                packed[name] = jnp.array(pars[name])

        return packed
