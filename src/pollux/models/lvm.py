from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
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


def select_outputs(
    spec: bool | Sequence[str] | Mapping[str, float] | None,
    names: Iterable[str],
    what: str,
) -> dict[str, float | None]:
    """Resolve an architecture's per-output selector into ``{name: value}``.

    Architectures take options that apply to some subset of their outputs. Rather
    than one argument per output, they take a single selector: ``True`` for every
    output, ``False`` for none, a sequence of names, or a mapping from name to a
    scalar setting.

    Parameters
    ----------
    spec
        The selector: ``True``, ``False``/``None``, a sequence of output names, or a
        mapping from output name to a scalar.
    names
        The output names the selector is allowed to mention.
    what
        What is being selected, used in the error message for unknown names.

    Returns
    -------
    dict
        The selected output names, mapped to the given scalar or to ``None`` where
        no scalar was supplied.

    Examples
    --------
    >>> from pollux.models.lvm import select_outputs
    >>> select_outputs(True, ["label", "flux"], "scatter")
    {'label': None, 'flux': None}
    >>> select_outputs(["flux"], ["label", "flux"], "scatter")
    {'flux': None}
    >>> select_outputs({"flux": 5.0}, ["label", "flux"], "scatter")
    {'flux': 5.0}
    >>> select_outputs(False, ["label", "flux"], "scatter")
    {}
    """
    names = list(names)

    if spec is True:
        return dict.fromkeys(names)
    if spec is False or spec is None:
        return {}

    selected: dict[str, float | None] = (
        dict(spec) if isinstance(spec, Mapping) else dict.fromkeys(spec)
    )

    unknown = [name for name in selected if name not in names]
    if unknown:
        msg = (
            f"{what} names {sorted(unknown)} are not outputs of this model. "
            f"Expected some of: {names}"
        )
        raise ValueError(msg)

    return selected


class LVMOutput(eqx.Module):
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


class LVM(eqx.Module):
    """A generative latent variable model with multiple outputs.

    This is the general framework the rest of the package is built on: each object is
    represented by a latent vector, and every observed output is generated as a
    transformation away from that vector. The outputs can be heterogeneous -- spectra,
    photometry, stellar labels, or anything else measured for the same objects -- and
    each one carries its own transform from the latents, plus an optional transform of
    its reported errors. The observed data are taken to be Gaussian about the predicted
    values. While the framework is general, it was written with stellar spectroscopic
    data in mind.

    Build a model by constructing it with a latent size and calling
    :meth:`register_output` once per output, then fit it with :meth:`optimize` or
    :meth:`optimize_iterative`.

    Specific architectures ship as subclasses that register their own outputs:
    :class:`~pollux.models.Lux` and :class:`~pollux.models.Cannon`. Reach for those when
    one of them describes your model, and for ``LVM`` directly when you want to compose
    the transforms yourself.

    Parameters
    ----------
    latent_size : int
        The size of the latent vector representation for each object (i.e. the embedded
        dimensionality).

    Examples
    --------
    >>> import pollux as plx
    >>> from pollux.models.transforms import LinearTransform, ScatterTransform
    >>> model = plx.LVM(latent_size=4)
    >>> model.register_output("label", LinearTransform(output_size=3))
    >>> model.register_output(
    ...     "flux",
    ...     LinearTransform(output_size=128),
    ...     err_transform=ScatterTransform(output_size=128),
    ... )
    >>> sorted(model.outputs)
    ['flux', 'label']

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
    # Not init=False: architectures register their outputs from inside __init__, so by
    # the time equinox inspects the instance this holds transform parameters -- and it
    # warns about array leaves reached through an init=False field.
    outputs: dict[str, LVMOutput] = eqx.field(default_factory=dict)

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
        self.outputs[name] = LVMOutput(data_transform, err_transform)

    def predict_outputs(
        self,
        pars: dict[str, Any],
        latents: BatchedLatentsT | None = None,
        names: list[str] | str | None = None,
    ) -> dict[str, BatchedOutputT]:
        """Predict output values for given parameters and latent vectors.

        Parameters
        ----------
        pars
            A dictionary of parameters for each output transformation in the model,
            in the nested format returned by :meth:`optimize`::

                {
                    "output_name": {
                        "data": {...} or [...],  # Transform parameters
                        "err": {...}             # Error transform parameters
                    },
                    "latents": array  # Used when ``latents`` is not passed
                }

            For single transforms, ``"data"`` is a dict: ``{"A": array, "b": array}``

            For :class:`TransformSequence`, ``"data"`` is a tuple of dicts:
            ``({"A": array}, {"b": array})``

        latents
            The latent vectors that transform into the outputs, with shape
            ``(n_objects, latent_size)``. If not passed, the latents are read from
            ``pars["latents"]`` -- which is where :meth:`optimize` puts them, so a
            round trip needs nothing but the optimized parameters. Pass this
            explicitly to predict for one set of objects using output parameters
            fitted on another, as when applying a trained model to a test set.
        names
            A single string or a list of output names to predict. If ``None``, predict
            all outputs (default).

        Returns
        -------
        dict
            A dictionary of predicted output values, where the keys are the output
            names and values are arrays of shape ``(n_objects, output_size)``.

        Examples
        --------
        Predict for the objects the model was optimized against::

            pars, _ = model.optimize(train_data, num_steps=1000, rng_key=key)
            predictions = model.predict_outputs(pars)

        Predict for a test set, holding the output parameters at their trained
        values::

            predictions = model.predict_outputs(
                model.output_pars(pars), latents=test_pars["latents"]
            )
        """
        # The argument order used to be (latents, pars). Stale positional calls would
        # otherwise reach the parameter walk below and fail somewhere confusing.
        if not isinstance(pars, Mapping):
            msg = (
                f"Expected a dict of parameters as the first argument, got "
                f"{type(pars).__name__}. predict_outputs() takes the parameters "
                "first and the latents second: predict_outputs(pars) reads the "
                "latents from pars['latents'], or pass "
                "predict_outputs(pars, latents=...) explicitly."
            )
            raise TypeError(msg)

        if latents is None:
            if "latents" not in pars:
                msg = (
                    "No latents given and none found in pars['latents']. Either pass "
                    "latents=... explicitly, or pass the parameters returned by "
                    "optimize() / optimize_iterative(), which include the latents."
                )
                raise KeyError(msg)
            latents = pars["latents"]

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

        data_pars = {}
        for name in names:
            # A transform with no learnable parameters -- NoOpTransform,
            # PolyFeatureTransform -- never appears in a parameter dict, so a missing
            # entry means "this output has no parameters", not "the caller forgot one"
            output_pars = pars.get(name, {})
            if not isinstance(output_pars, dict):
                msg = (
                    f"Expected dict for parameters of output '{name}', "
                    f"got {type(output_pars).__name__}"
                )
                raise TypeError(msg)

            if output_pars and "data" not in output_pars and "err" not in output_pars:
                msg = (
                    f"Parameters for output '{name}' have no 'data' or 'err' key. "
                    "Transform parameters must be nested under 'data' (and error "
                    f"transform parameters under 'err'), as in "
                    f"{{'{name}': {{'data': {{'A': ...}}}}}} -- which is the format "
                    "optimize() returns."
                )
                raise ValueError(msg)

            data_pars[name] = output_pars.get("data", {})

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

    @staticmethod
    def output_pars(pars: dict[str, Any]) -> dict[str, Any]:
        """Return the parameters that are shared across objects, i.e. everything
        but the latents.

        The latents are per-object, so they are exactly the parameters that do not
        carry over from a training set to a test set. This is the dict to pass as
        ``fixed_pars`` when applying a trained model to new objects.

        Parameters
        ----------
        pars
            A parameter dictionary, as returned by :meth:`optimize` or
            :meth:`optimize_iterative`.

        Returns
        -------
        dict
            A shallow copy of ``pars`` without the ``"latents"`` entry.

        Examples
        --------
        >>> pars = {"latents": 1, "flux": {"data": {"A": 2}}}
        >>> LVM.output_pars(pars)
        {'flux': {'data': {'A': 2}}}
        """
        return {k: v for k, v in pars.items() if k != "latents"}

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
            The latent vectors that transform into the outputs. Whether these are
            inferred or known depends on the architecture: in :class:`~pollux.models.Lux`
            they are free parameters, whereas in :class:`~pollux.models.Cannon` they are
            the observed stellar labels of the training set.
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
        outputs = self.predict_outputs(nested_pars, latents, names=output_names)
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
        """Create the default numpyro model for this LVM model.

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

        Wherever a sub-problem turns out to be quadratic, this solves it exactly with
        weighted least squares instead of running SVI on it. Whether it is quadratic
        is established by linearizing the transform, so composed models count too --
        a slice of the latents feeding a linear branch, polynomial features feeding a
        linear layer, a linear map plus a fixed offset. The default strategy
        alternates between optimizing the latents (with output parameters fixed) and
        optimizing each output's parameters (with the latents fixed).

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
