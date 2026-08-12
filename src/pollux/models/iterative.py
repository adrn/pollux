"""Iterative optimization strategies for Lux.

This module provides an alternating/block coordinate descent optimization scheme
that exploits the structure of the Lux model for faster convergence.
"""

from __future__ import annotations

__all__ = [
    "IterativeOptimizationResult",
    "ParameterBlock",
    "optimize_iterative",
]

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from tqdm.auto import tqdm
from typing_extensions import TypeIs

from .._linalg import weighted_least_squares
from ..data import OutputData, PolluxData
from .transforms import (
    AbstractTransform,
    AffineTransform,
    LinearTransform,
    OffsetTransform,
    TransformSequence,
)

if TYPE_CHECKING:
    from .lux import Lux

#: Transforms whose least squares sub-problem has a closed-form solution
type LinearTransformT = LinearTransform | AffineTransform | OffsetTransform


@dataclass
class ParameterBlock:
    """Specification for a block of parameters to optimize together.

    This allows fine-grained control over which parameters are optimized in each
    iteration step and how they are optimized.

    Parameters
    ----------
    name
        Name of the parameter block (for logging and identification).
    params
        Which parameters to include. Can be:
        - ``"latents"``: Optimize latent vectors
        - ``"output_name"``: Optimize all parameters for a specific output
        - ``"output_name:data"``: Optimize only data transform parameters
        - ``"output_name:err"``: Optimize only error transform parameters
    optimizer
        The optimizer to use for this block. If ``"least_squares"``, uses a closed-form
        weighted least squares solution (only valid for linear models).
        If None, uses numpyro SVI with ``numpyro.optim.Adam`` at ``step_size=1e-3``
        by default. Pass a different optimizer class and/or set ``optimizer_kwargs``
        to override.
    optimizer_kwargs
        Keyword arguments to pass to the optimizer constructor. When ``optimizer``
        is None (i.e., Adam is used), the default is ``{"step_size": 1e-3}``; any
        keys provided here override that default. Ignored when using
        ``"least_squares"``.
    num_steps
        Number of optimization steps for this block (for SVI optimizers).
        Ignored for least_squares.

    Examples
    --------
    Optimize latents with least squares (fast, closed-form):

    >>> latent_block = ParameterBlock(
    ...     name="latents",
    ...     params="latents",
    ...     optimizer="least_squares",
    ... )

    Optimize flux parameters with Adam and custom learning rate:

    >>> flux_block = ParameterBlock(  # doctest: +SKIP
    ...     name="flux",
    ...     params="flux:data",
    ...     optimizer=numpyro.optim.Adam,
    ...     optimizer_kwargs={"step_size": 1e-3},
    ...     num_steps=1000,
    ... )

    """

    name: str
    params: str | list[str]
    optimizer: Literal["least_squares"] | type | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    num_steps: int = 1000

    @property
    def params_list(self) -> list[str]:
        """``params`` as a list, whether one spec or several were given."""
        return [self.params] if isinstance(self.params, str) else self.params


@dataclass
class IterativeOptimizationResult:
    """Result of iterative optimization.

    Parameters
    ----------
    params
        The optimized parameters in unpacked format.
    losses_per_cycle
        List of loss values at the end of each cycle.
    n_cycles
        Number of full cycles completed.
    converged
        Whether the optimization converged according to tolerance.

    """

    params: dict[str, Any]
    losses_per_cycle: list[float]
    n_cycles: int
    converged: bool


def _is_linear_transform(transform: Any) -> TypeIs[LinearTransformT]:
    """Check if a transform is linear (amenable to least squares).

    Note: TransformSequence is not supported for iterative optimization,
    even if all component transforms are linear.
    """
    return isinstance(transform, (LinearTransform, AffineTransform, OffsetTransform))


def _split_param_layer(
    transform: AbstractTransform,
) -> tuple[TransformSequence | None, LinearTransform | AffineTransform] | None:
    """Split a transform into a parameter-free prefix and a trailing linear layer.

    The per-output-dimension least squares solve needs two things: the features
    arriving at the linear layer, and the layer itself. Running the prefix forward
    supplies the features, so it does not matter what the prefix *is* -- a latent
    slice, polynomial features, anything -- as long as it carries no parameters of
    its own to solve for.

    Returns ``(prefix, layer)``, with ``prefix`` None when the transform is already
    a bare linear layer, or None when the transform cannot be solved this way.
    """
    if isinstance(transform, (LinearTransform, AffineTransform)):
        return None, transform

    if not isinstance(transform, TransformSequence):
        return None

    *head, last = transform.transforms
    if not head or not isinstance(last, (LinearTransform, AffineTransform)):
        return None

    # Parameters anywhere but the last layer would be silently left unoptimized
    if any(transform.names_nested[:-1]):
        return None

    return TransformSequence(tuple(head)), last


#: Relative tolerance for deciding that a prediction map is affine in the latents.
#: A composition of linear primitives reproduces its own linearization bitwise, while
#: a nonlinearity contributing 1e-4 of the signal shows up at ~1e-4, so anything in
#: 1e-6..1e-5 separates the two by orders of magnitude.
_AFFINE_RTOL = 1e-6

#: Types for the two closures :func:`jax.linearize` and :func:`jax.linear_transpose`
#: hand back: apply the effective design matrix, and apply its transpose.
type JVPFuncT = Callable[[jax.Array], jax.Array]
type VJPFuncT = Callable[[jax.Array], tuple[jax.Array, ...]]


def _linearize_latents(
    f: JVPFuncT, z0: jax.Array, probe: jax.Array
) -> tuple[jax.Array, JVPFuncT, VJPFuncT] | None:
    """Linearize a prediction map in the latents, or return None if it is not affine.

    Returns ``(c, jvp, vjpT)`` with ``f(z) == jvp(z) + c``, where ``jvp`` applies the
    effective design matrix ``J`` and ``vjpT`` applies its transpose.

    This is exact rather than approximate. For a composition of linear primitives the
    JVP runs the same multiply-add sequence on the same values as the primal, so it
    returns the design matrix itself: for a bare :class:`~pollux.models.transforms.
    LinearTransform` it recovers ``A`` bitwise, and ``c`` is bitwise zero. Whether
    the map really is affine is not assumed, it is measured -- hence the probe.

    ``J`` is never materialized. For spectra-sized outputs it would be ``latent_size``
    copies of the data; going through ``jvp``/``vjpT`` keeps every temporary the size
    of one output array.

    Parameters
    ----------
    f
        Maps latents of shape ``(n_data, latent_size)`` to predictions of shape
        ``(n_data, output_size)``. Each object's prediction must depend only on its
        own latents -- the same plate independence the numpyro model assumes. The
        transpose sums over objects, so a genuinely cross-object map would fold the
        cross terms into the wrong place.
    z0
        Zeros of the latent shape; the point to linearize about. Affine maps have the
        same derivative everywhere, so the choice only fixes ``c = f(0)``.
    probe
        Where to test affineness, ideally at the amplitude of the real latents: the
        residual of a nonlinear map grows with the probe amplitude.
    """
    c, jvp = jax.linearize(f, z0)
    pred = f(probe)
    residual = jnp.abs(pred - (c + jvp(probe))).max()
    if residual > _AFFINE_RTOL * jnp.abs(pred).max():
        return None
    return c, jvp, jax.linear_transpose(jvp, z0)


def _latents_probe_point(
    latents: jax.Array | None, shape: tuple[int, ...]
) -> jax.Array:
    """A point to test affineness at, scaled to the latents we are actually fitting."""
    scale = 1.0 if latents is None else jnp.maximum(jnp.abs(latents).max(), 1.0)
    # Fixed key: the affineness verdict should not depend on when it is asked.
    return scale * jax.random.normal(jax.random.PRNGKey(0), shape)


def _inverse_variance(output_data: OutputData) -> jax.Array:
    """Inverse-variance weights for an output, or ones where errors are absent."""
    err = output_data.err
    if jnp.all(err <= 0):
        return jnp.ones_like(output_data.data)
    return 1.0 / err**2


def _get_regularization_from_prior(
    prior: dist.Distribution,
) -> tuple[jax.Array | float, jax.Array | float]:
    """Extract regularization parameters from a prior distribution.

    Parameters
    ----------
    prior
        A numpyro distribution. For Normal distributions, extracts the precision.
        For other distributions, returns a negligible fallback regularization.

    Returns
    -------
    regularization
        The regularization strength alpha = 1 / scale**2.
    prior_mean
        The prior mean μ (for regularization toward non-zero mean).

    Notes
    -----
    Currently only supports Normal distributions. For other priors,
    uses a negligible regularization with zero mean.
    """
    if isinstance(prior, dist.Normal):
        # Normal(loc, scale): regularization is 1/scale^2
        scale = prior.scale
        loc = prior.loc
        return 1.0 / (scale**2), loc
    if isinstance(prior, dist.ImproperUniform):
        # No regularization for improper uniform
        return 0.0, 0.0
    # Fallback for other distributions
    return 1e-6, 0.0


def _solve_latents_least_squares(
    model: Lux,
    data: PolluxData,
    current_params: dict[str, Any],
    latents_prior: dist.Distribution | None = None,
) -> jax.Array:
    """Solve for optimal latents using weighted least squares.

    Each output contributes a prediction ``y ≈ J z + c``, where ``J`` is whatever
    effective design matrix the output's transform amounts to. Summing the outputs'
    contributions gives one normal-equation system per object::

        (sum_o Jo^T Wo Jo + λI) z = sum_o Jo^T Wo (yo - co) + λ μ

    ``J`` and ``c`` come from linearizing the transform (see
    :func:`_linearize_latents`), not from looking up a parameter by name, so any
    composition that happens to be affine in the latents works: a slice feeding a
    linear map, a :class:`~pollux.models.transforms.ConcatenateTransform` of linear
    children, a linear map plus a fixed per-object offset. Compositions that are not
    affine, like polynomial features of the latents, are rejected here and belong in
    an SVI block.

    Parameters
    ----------
    model
        The Lux instance.
    data
        The data to fit.
    current_params
        Current parameter estimates. Everything except the latents is held fixed.
    latents_prior
        Prior distribution for latents. If None, uses Normal(0, 1).
        The regularization strength is extracted from this prior.

    Returns
    -------
    latents
        Optimal latent vectors of shape (n_stars, latent_size).

    Notes
    -----
    Memory: no design matrix is ever formed. The largest temporaries are one output
    array, ``(n_data, output_size)``, and the accumulated ``(n_data, latent_size,
    latent_size)`` system -- the same footprint as the explicit-``A`` version this
    replaces, and independent of how the transform is composed.

    """
    n_data = len(data)
    latent_size = model.latent_size

    z0 = jnp.zeros((n_data, latent_size))
    probe = _latents_probe_point(current_params.get("latents"), z0.shape)
    basis = jnp.eye(latent_size)

    # Sum the per-output contributions to the normal equations
    AtWA = jnp.zeros((n_data, latent_size, latent_size))
    AtWy = jnp.zeros((n_data, latent_size))

    for output_name in model.outputs:
        if output_name not in data:
            continue

        # Linearize exactly what the model predicts, so the solver cannot drift
        # away from predict_outputs
        def predict(z: jax.Array, name: str = output_name) -> jax.Array:
            return model.predict_outputs(z, current_params, names=[name])[name]

        linearized = _linearize_latents(predict, z0, probe)
        if linearized is None:
            msg = (
                f"Output '{output_name}' is not affine in the latents, so the latents "
                "cannot be solved in closed form. Optimize them with an SVI block "
                "instead (ParameterBlock('latents', 'latents'))."
            )
            raise ValueError(msg)
        c, jvp, vjpT = linearized

        output_data = data[output_name]
        w = _inverse_variance(output_data)

        # J^T W (y - c), one VJP
        AtWy = AtWy + vjpT(w * (output_data.data - c))[0]

        # J^T W J column by column: push a basis vector through J, weight it, pull it
        # back. Column k costs one JVP and one VJP, and no (n, output_size, latent_size)
        # intermediate is built.
        AtWA = AtWA + jnp.stack(
            [vjpT(w * jvp(jnp.broadcast_to(e, z0.shape)))[0] for e in basis], axis=-1
        )

    # Get regularization from latents prior
    if latents_prior is None:
        latents_prior = dist.Normal(0.0, 1.0)
    reg_strength, prior_mean = _get_regularization_from_prior(latents_prior)

    # Add regularization: (A^T W A + λI) z = A^T W y + λ μ
    # For N(0, 1) prior, this reduces to (A^T W A + I) z = A^T W y
    reg_matrix = reg_strength * jnp.eye(latent_size)
    AtWA = AtWA + reg_matrix[None, :, :]

    # Add prior mean contribution to RHS if non-zero
    if not jnp.allclose(prior_mean, 0.0):
        AtWy = AtWy + reg_strength * prior_mean

    # Solve for each data point: z[i] = solve(AtWA[i], AtWy[i])
    result: jax.Array = jax.vmap(jnp.linalg.solve)(AtWA, AtWy)
    return result


def _solve_output_params_least_squares(
    model: Lux,
    data: PolluxData,
    output_name: str,
    latents: jax.Array,
) -> dict[str, Any] | tuple[dict[str, Any], ...]:
    """Solve for optimal output parameters using weighted least squares.

    The output is modelled as a linear layer sitting on features ``X`` derived from
    the latents, ``y = A @ X + b``. Each output dimension j is then an independent
    small problem,

        A[j, :] = (X^T W_j X + λI)^{-1} X^T W_j y[:, j]

    with ``W_j = diag(1/err[:, j]^2)``, which is what makes this fast: ``output_size``
    independent ``n_features``-sized solves rather than one big system.

    ``X`` is whatever reaches the linear layer, not necessarily the latents: for a
    bare :class:`~pollux.models.transforms.LinearTransform` it *is* the latents, but
    for a sequence it is the prefix run forward -- a latent slice, or the polynomial
    expansion that makes this the Cannon. A bias term, if the layer has one, is
    solved jointly as an extra column of ones.

    The regularization strengths λ come from the layer's priors on ``A`` and ``b``.

    Parameters
    ----------
    model
        The Lux instance.
    data
        The data to fit.
    output_name
        Name of the output to optimize.
    latents
        Current latent vectors of shape (n_data, latent_size).

    Returns
    -------
    params
        Optimized parameters for this output, in the nested format the transform
        expects: a dict for a bare layer, a tuple of per-child dicts for a sequence.

    """
    transform = model.outputs[output_name].data_transform
    split = _split_param_layer(transform)
    if split is None:
        msg = (
            f"Output '{output_name}' does not end in a linear layer with all of its "
            "parameters in that layer, so its parameters cannot be solved in closed "
            "form. Optimize this output with an SVI block instead."
        )
        raise ValueError(msg)
    prefix, layer = split

    if output_name not in data:
        msg = f"No data found for output '{output_name}'"
        raise ValueError(msg)

    output_data = data[output_name]
    y = output_data.data  # (n_data, output_size)
    output_ivar = _inverse_variance(output_data)
    output_size = y.shape[1]

    # Features arriving at the linear layer. A parameter-free prefix takes no
    # arguments, which is exactly what _split_param_layer guarantees.
    features = latents if prefix is None else prefix.apply(latents)
    has_bias = "b" in layer.shapes
    design = (
        jnp.concatenate([features, jnp.ones((len(y), 1))], axis=1)
        if has_bias
        else features
    )
    n_design = design.shape[1]

    # Regularization from the layer's own priors; the bias column gets its own entry
    alpha, mu = _get_regularization_from_prior(layer.priors.get("A", dist.Normal(0, 1)))
    reg_matrix = alpha * jnp.eye(n_design)
    rhs_extra = alpha * jnp.broadcast_to(mu, (output_size, n_design))
    if has_bias:
        alpha_b, mu_b = _get_regularization_from_prior(
            layer.priors.get("b", dist.Normal(0, 1))
        )
        reg_matrix = reg_matrix.at[-1, -1].set(alpha_b)
        rhs_extra = rhs_extra.at[:, -1].set(alpha_b * mu_b)

    def fit_single_output_dim(
        y_dim: jax.Array, ivar_dim: jax.Array, rhs_row: jax.Array
    ) -> jax.Array:
        """Fit the features -> output coefficients for a single output dimension."""
        return weighted_least_squares(design, y_dim, ivar_dim, reg_matrix, rhs_row)

    solution: jax.Array = jax.vmap(fit_single_output_dim)(
        y.T, output_ivar.T, rhs_extra
    )  # (output_size, n_design)

    solved = (
        {"A": solution[:, :-1], "b": solution[:, -1]} if has_bias else {"A": solution}
    )
    if prefix is None:
        return solved

    # Name the parameters for their position in the sequence, then let the transform
    # put them back into its own nested layout
    last = len(prefix.transforms)
    return transform.unpack_pars(
        {f"{last}:{name}": value for name, value in solved.items()}, ignore_missing=True
    )


def _string_to_parameter_block(model: Lux, name: str) -> ParameterBlock:
    """Convert a string block name to a ParameterBlock with inferred optimizer."""
    optimizer: Literal["least_squares"] | None
    if name == "latents":
        optimizer = "least_squares" if _all_outputs_linear(model) else None
        return ParameterBlock(name="latents", params="latents", optimizer=optimizer)

    output_name = name.split(":", maxsplit=1)[0]
    if output_name not in model.outputs:
        msg = f"Unknown parameter block: '{name}'"
        raise ValueError(msg)

    transform = model.outputs[output_name].data_transform
    optimizer = "least_squares" if _is_linear_transform(transform) else None
    return ParameterBlock(name=name, params=name, optimizer=optimizer)


def _build_initial_params_from_fixed(
    model: Lux,
    data: PolluxData,
    fixed_pars: dict[str, Any],
    blocks: list[ParameterBlock],
) -> dict[str, Any]:
    """Build initial params by merging fixed_pars with zero-initialized optimized params."""
    initial: dict[str, Any] = dict(fixed_pars)

    if "latents" not in initial and any("latents" in b.params_list for b in blocks):
        initial["latents"] = jnp.zeros((len(data), model.latent_size))

    return initial


def optimize_iterative(
    model: Lux,
    data: PolluxData,
    blocks: list[ParameterBlock] | list[str] | None = None,
    fixed_pars: dict[str, Any] | None = None,
    max_cycles: int = 100,
    tol: float = 1e-4,
    rng_key: jax.Array | None = None,
    initial_params: dict[str, Any] | None = None,
    latents_prior: dist.Distribution | None = None,
    progress: bool = True,
) -> IterativeOptimizationResult:
    """Optimize model using iterative block coordinate descent.

    This implements an alternating optimization strategy that cycles through
    parameter blocks, optimizing each while holding others fixed. For linear
    models, each sub-problem can be solved exactly using weighted least squares.

    The default strategy alternates between:
    1. Optimize latents (with output parameters fixed)
    2. Optimize each output's parameters (with latents and other outputs fixed)

    Parameters
    ----------
    model
        The Lux to optimize.
    data
        The training data.
    blocks
        List of :class:`ParameterBlock` specifications, or a list of strings
        naming which parameter groups to optimize (e.g. ``["latents"]``).
        If strings are given, :class:`ParameterBlock` instances are constructed
        automatically with an inferred optimizer (``"least_squares"`` for linear
        transforms). If None, uses a default strategy that alternates between
        latents and each output.
    fixed_pars
        Parameters to hold fixed during optimization. When provided alongside
        string ``blocks``, the function initializes latents to zero and merges
        ``fixed_pars`` with the optimized parameters before returning, so the
        result contains a complete parameter dict. Ignored when ``initial_params``
        is also provided (caller is responsible for merging in that case).
    max_cycles
        Maximum number of full optimization cycles.
    tol
        Convergence tolerance. Stops when relative change in loss < tol.
    rng_key
        JAX random key. Required when any block uses SVI (i.e., ``optimizer !=
        "least_squares"``) or when ``initial_params`` is None (used to sample
        initial values from the model priors; falls back to
        ``jax.random.PRNGKey(0)`` if not provided in that case).
    initial_params
        Initial parameter values. If None and ``fixed_pars`` is provided, built
        automatically by merging ``fixed_pars`` with zero-initialized optimized
        params. If both are None, initialized from priors.
    latents_prior
        Prior distribution for latents. If None, uses Normal(0, 1).
        Used to determine regularization strength for latent least squares.
    progress
        Whether to display a tqdm progress bar showing optimization progress.

    Returns
    -------
    IterativeOptimizationResult
        The optimization result containing optimized parameters and convergence
        info. When ``fixed_pars`` is provided, ``result.params`` includes both
        the fixed and optimized parameters.

    Notes
    -----
    When a block has ``optimizer=None``, SVI is run with ``numpyro.optim.Adam``
    at ``step_size=1e-3``. Override via ``optimizer_kwargs`` on the block, e.g.
    ``ParameterBlock(..., optimizer_kwargs={"step_size": 1e-4})``.

    Examples
    --------
    Basic usage with default blocks:

    >>> result = optimize_iterative(model, data, max_cycles=20)  # doctest: +SKIP
    >>> opt_params = result.params  # doctest: +SKIP

    Custom block specification:

    >>> blocks = [  # doctest: +SKIP
    ...     ParameterBlock("latents", "latents", optimizer="least_squares"),
    ...     ParameterBlock("flux", "flux:data", optimizer="least_squares"),
    ...     ParameterBlock("labels", "label:data", num_steps=500),
    ... ]
    >>> result = optimize_iterative(model, data, blocks=blocks)  # doctest: +SKIP

    Optimizing only latents with fixed output parameters (e.g. applying a
    trained model to new test data):

    >>> result = optimize_iterative(  # doctest: +SKIP
    ...     model, test_data, blocks=["latents"], fixed_pars=trained_pars
    ... )
    >>> test_opt_pars = result.params  # already contains fixed + optimized  # doctest: +SKIP

    """
    # Resolve blocks to list[ParameterBlock] | None, converting any string specs.
    # Done per element rather than by sniffing blocks[0], so a mixed list works.
    _blocks: list[ParameterBlock] | None = (
        None
        if blocks is None
        else [
            _string_to_parameter_block(model, b) if isinstance(b, str) else b
            for b in blocks
        ]
    )

    # Build initial_params from fixed_pars if not provided
    if initial_params is None and fixed_pars is not None:
        initial_params = _build_initial_params_from_fixed(
            model, data, fixed_pars, _blocks or []
        )

    # Default blocks: alternate between latents and each output
    if _blocks is None:
        _blocks = [
            _string_to_parameter_block(model, name)
            for name in ("latents", *(f"{o}:data" for o in model.outputs))
        ]

    # Warn if any output has err_transform parameters that are neither being
    # optimized (in active blocks) nor intentionally held fixed (in fixed_pars)
    active_block_params = {b.params for b in _blocks}
    for output_name, lux_output in model.outputs.items():
        err_key = f"{output_name}:err"
        err_is_fixed = (
            fixed_pars is not None
            and output_name in fixed_pars
            and "err" in fixed_pars[output_name]
        )
        if err_key not in active_block_params and not err_is_fixed:
            et = lux_output.err_transform
            priors = et.priors
            has_params = (
                any(len(p) > 0 for p in priors)
                if isinstance(priors, tuple)
                else len(priors) > 0
            )
            if has_params:
                warnings.warn(
                    f"Output '{output_name}' has an err_transform with learnable "
                    f"parameters, but '{err_key}' is not in the active optimization "
                    "blocks. These parameters will not be updated during iterative "
                    f"optimization. To optimize them, add a ParameterBlock with "
                    f"params='{err_key}'.",
                    UserWarning,
                    stacklevel=2,
                )

    # Initialize parameters by sampling from priors
    if initial_params is None:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        rng_key, init_key = jax.random.split(rng_key)
        predictive = Predictive(model.default_numpyro_model, num_samples=1)
        packed_samples = predictive(init_key, data)
        # Remove the batch dimension from num_samples=1, and filter out
        # observed samples (keys starting with "obs:")
        packed_samples = {
            k: v[0] for k, v in packed_samples.items() if not k.startswith("obs:")
        }
        current_params = model.unpack_numpyro_pars(packed_samples)
    else:
        current_params = initial_params

    losses_per_cycle: list[float] = []

    prev_loss = float("inf")

    # Set up progress bar
    pbar = tqdm(
        range(max_cycles),
        desc="Iterative optimization",
        disable=not progress,
    )

    n_cycles = 0
    converged = False

    for cycle in pbar:
        n_cycles = cycle + 1

        for block in _blocks:
            if block.optimizer == "least_squares":
                current_params = _optimize_block_least_squares(
                    model, data, block, current_params, latents_prior
                )
            else:
                # Use numpyro SVI for non-linear blocks
                if rng_key is None:
                    msg = "rng_key required for SVI-based optimization"
                    raise ValueError(msg)
                rng_key, subkey = jax.random.split(rng_key)
                current_params = _optimize_block_numpyro(
                    model, data, block, current_params, subkey, latents_prior
                )

        # Compute loss at end of cycle
        loss = _compute_loss(model, data, current_params)
        losses_per_cycle.append(float(loss))

        # Update progress bar with loss info
        rel_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-8)
        pbar.set_postfix(
            loss=f"{loss:.4g}",
            rel_change=f"{rel_change:.2e}",
        )

        # Check convergence
        converged = bool(rel_change < tol)
        if converged:
            pbar.set_description("Converged")
            pbar.update(max_cycles - pbar.n)  # Complete the bar
            pbar.set_postfix(loss=f"{loss:.4g}")
            break
        prev_loss = loss

    pbar.colour = "green" if converged else "red"
    pbar.close()

    return IterativeOptimizationResult(
        params=current_params,
        losses_per_cycle=losses_per_cycle,
        n_cycles=n_cycles,
        converged=converged,
    )


def _all_outputs_linear(model: Lux) -> bool:
    """Check if all model outputs use linear transforms."""
    return all(
        _is_linear_transform(out.data_transform) for out in model.outputs.values()
    )


def _optimize_block_least_squares(
    model: Lux,
    data: PolluxData,
    block: ParameterBlock,
    current_params: dict[str, Any],
    latents_prior: dist.Distribution | None = None,
) -> dict[str, Any]:
    """Optimize a parameter block using least squares."""
    new_params = dict(current_params)

    for param_spec in block.params_list:
        if param_spec == "latents":
            new_params["latents"] = _solve_latents_least_squares(
                model, data, current_params, latents_prior
            )
            continue

        # An output name, optionally qualified: only "data" params are solvable here
        output_name, _, param_type = param_spec.partition(":")
        if param_type not in ("", "data"):
            continue

        if output_name not in new_params:
            new_params[output_name] = {"data": {}, "err": {}}
        new_params[output_name]["data"] = _solve_output_params_least_squares(
            model, data, output_name, current_params["latents"]
        )

    return new_params


def _optimize_block_numpyro(
    model: Lux,
    data: PolluxData,
    block: ParameterBlock,
    current_params: dict[str, Any],
    rng_key: jax.Array,
    latents_prior: dist.Distribution | None = None,
) -> dict[str, Any]:
    """Optimize a parameter block using numpyro SVI.

    This function optimizes a subset of parameters (specified in the block)
    while holding all other parameters fixed. It uses numpyro's SVI with
    AutoDelta guide for MAP estimation.

    Parameters
    ----------
    model
        The Lux instance.
    data
        The training data.
    block
        ParameterBlock specification including which parameters to optimize,
        the optimizer to use, and the number of optimization steps.
    current_params
        Current parameter estimates (unpacked format). Parameters not being
        optimized will be held fixed.
    rng_key
        JAX random key for SVI.
    latents_prior
        Prior distribution for latents. If None, uses Normal(0, 1).

    Returns
    -------
    dict
        Updated parameters with the optimized block values merged in.

    Notes
    -----
    The optimizer defaults to Adam with step_size=1e-3 if not specified
    in the block.
    """
    # Build fixed_pars containing everything NOT being optimized
    fixed_pars = _build_fixed_pars(model, current_params, block.params_list)

    # Build the optimizer
    optimizer_cls = block.optimizer
    if optimizer_cls is None:
        optimizer_cls = numpyro.optim.Adam
    elif optimizer_cls == "least_squares":
        msg = (
            "Least squares optimization should be handled by "
            "_optimize_block_least_squares"
        )
        raise ValueError(msg)

    optimizer_kwargs = {"step_size": 1e-3, **block.optimizer_kwargs}
    optimizer = optimizer_cls(**optimizer_kwargs)

    # Pack fixed parameters for numpyro
    packed_fixed_pars = model.pack_numpyro_pars(fixed_pars, ignore_missing=True)

    # Create partial model with fixed parameters (names=None: all outputs)
    partial_model = partial(
        model.default_numpyro_model,
        fixed_pars=packed_fixed_pars,
        latents_prior=latents_prior,
    )

    # Run SVI optimization
    svi_key, sample_key = jax.random.split(rng_key)
    guide = AutoDelta(partial_model)
    svi = SVI(partial_model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(svi_key, block.num_steps, data, progress_bar=False)

    # Extract optimized parameters
    packed_map_pars = guide.sample_posterior(sample_key, svi_results.params)
    optimized_subset = model.unpack_numpyro_pars(packed_map_pars, ignore_missing=True)

    # Merge optimized parameters with current parameters
    new_params = dict(current_params)
    for param_spec in block.params_list:
        if param_spec == "latents" and "latents" in optimized_subset:
            new_params["latents"] = optimized_subset["latents"]
        elif ":" in param_spec:
            output_name, param_type = param_spec.split(":", 1)
            if output_name in optimized_subset:
                if output_name not in new_params:
                    new_params[output_name] = {"data": {}, "err": {}}
                opt_output = optimized_subset[output_name]
                if param_type == "data" and "data" in opt_output:
                    new_params[output_name]["data"] = opt_output["data"]
                elif param_type == "err" and "err" in opt_output:
                    new_params[output_name]["err"] = opt_output["err"]
        elif param_spec in optimized_subset:
            new_params[param_spec] = optimized_subset[param_spec]

    return new_params


def _build_fixed_pars(
    model: Lux,
    current_params: dict[str, Any],
    optimize_params: list[str],
) -> dict[str, Any]:
    """Build fixed_pars dict containing everything not being optimized."""
    fixed: dict[str, Any] = {}

    # Check if latents should be fixed
    if "latents" not in optimize_params:
        fixed["latents"] = current_params.get("latents")

    # Check each output: hold "data" / "err" fixed unless this block optimizes them
    for output_name in model.outputs:
        output_params = current_params.get(output_name, {})
        output_fixed = {
            key: output_params[key]
            for key in ("data", "err")
            if key in output_params
            and output_name not in optimize_params
            and f"{output_name}:{key}" not in optimize_params
        }
        if output_fixed:
            fixed[output_name] = output_fixed

    return fixed


def _compute_loss(
    model: Lux,
    data: PolluxData,
    params: dict[str, Any],
) -> float:
    """Compute the negative log likelihood loss."""
    latents = params["latents"]
    predictions = model.predict_outputs(latents, params)

    total_loss = 0.0
    for output_name in model.outputs:
        if output_name not in data:
            continue

        output_data = data[output_name]
        pred = predictions[output_name]
        obs = output_data.data

        # Gaussian negative log likelihood (ignoring constant)
        chi2 = (pred - obs) ** 2 * _inverse_variance(output_data)
        total_loss = float(total_loss) + float(0.5 * jnp.sum(chi2))

    return total_loss
