"""Iterative optimization strategies for latent variable models.

This module provides an alternating/block coordinate descent optimization scheme
that exploits the structure of an LVM for faster convergence.
"""

from __future__ import annotations

__all__ = [
    "IterativeOptimizationResult",
    "ParameterBlock",
    "optimize_iterative",
]

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value
from tqdm.auto import tqdm

from .._linalg import weighted_least_squares
from ..data import PolluxData
from ..exceptions import PolluxLinearizationWarning
from .transforms import (
    AbstractTransform,
    AffineTransform,
    LinearTransform,
    TransformSequence,
)

if TYPE_CHECKING:
    from .lvm import LVM


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
    blocks
        The parameter blocks as they were actually run, after checking which ones
        could use a closed-form solve. Inspect ``block.optimizer`` to see what each
        block got: ``"least_squares"`` for the exact solve, anything else for SVI.

    """

    params: dict[str, Any]
    losses_per_cycle: list[float]
    n_cycles: int
    converged: bool
    blocks: list[ParameterBlock] = field(default_factory=list)


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


def _rows_are_independent(jvp: Callable[[jax.Array], jax.Array], shape: tuple) -> bool:
    """Whether each object's prediction depends only on its own latents.

    The per-object normal equations assume a block-diagonal Jacobian, one block per
    object. A transform written with ``vmap=False`` can quietly break that:
    ``z - z.mean(axis=0)`` is perfectly affine and passes every affineness probe, but
    couples every object to every other, and the solve would then return confident
    nonsense rather than refusing.

    Push a tangent supported on every other object through the Jacobian. If the rows
    are independent the response is supported on the same rows; any global reduction
    -- a mean, a sum, a normalization -- leaks outside them and is caught. A coupling
    between two objects that happen to fall on the same side of the split would slip
    through, but transforms couple objects globally or not at all in practice.
    """
    inside = (jnp.arange(shape[0]) % 2 == 0)[:, None]
    tangent = jnp.where(inside, jax.random.normal(jax.random.PRNGKey(2), shape), 0.0)
    response = jvp(tangent)
    leaked = jnp.abs(jnp.where(inside, 0.0, response)).max()
    return bool(leaked <= _AFFINE_RTOL * jnp.abs(response).max())


def _linearize_latents(
    f: Callable[[jax.Array], jax.Array],
    z0: jax.Array,
    probes: tuple[jax.Array, ...],
) -> tuple[jax.Array, Any, Any] | str:
    """Linearize a prediction map in the latents, or say why it cannot be used.

    Returns ``(c, jvp, vjpT)`` with ``f(z) == jvp(z) + c``, where ``jvp`` applies the
    effective design matrix ``J`` and ``vjpT`` applies its transpose, or a string
    describing why this map is not usable for a closed-form solve. ``J`` is never
    materialized: for spectra-sized outputs it would be ``latent_size`` copies of the
    data, while ``jvp``/``vjpT`` keep every temporary the size of one output array.

    :func:`jax.linearize` never fails -- handed a nonlinear function it returns the
    tangent plane, which would be silently wrong to use as a design matrix. So the
    verdict comes from testing the defining property, ``f(z) == f(0) + J z``, at the
    given probe points. That is one-sided: an affine map reproduces its linearization
    exactly and can never be wrongly refused, while a nonlinear one could in principle
    slip through by touching its tangent plane exactly where we look. See the
    "Closed-form solves, found by linearization" page in the documentation for why
    that is unlikely enough to rely on.

    Affine is necessary but not sufficient: the solve also needs each object's
    prediction to depend only on its own latents, which :func:`_rows_are_independent`
    checks. Being affine says nothing about that -- centering the latents across
    objects is affine and violates it.

    Note that both verdicts are properties of ``f``, which has the *current*
    parameters bound into it. A transform can be affine in the latents at one set of
    parameter values and not at another, so neither answer is settled for good.
    """
    c, jvp = jax.linearize(f, z0)

    for probe in probes:
        pred = f(probe)
        residual = jnp.abs(pred - (c + jvp(probe))).max()
        if residual > _AFFINE_RTOL * jnp.abs(pred).max():
            return "is not affine in the latents"

    if not _rows_are_independent(jvp, z0.shape):
        return "couples objects to one another, so there is no per-object solve to make"

    return c, jvp, jax.linear_transpose(jvp, z0)


def _output_predict_fn(
    model: LVM, output_name: str, params: dict[str, Any]
) -> Callable[[jax.Array], jax.Array]:
    """One output's latents -> prediction map, via predict_outputs so it cannot drift."""

    def predict(latents: jax.Array) -> jax.Array:
        return model.predict_outputs(params, latents, names=[output_name])[output_name]

    return predict


def _latents_probe_points(
    latents: jax.Array | None, shape: tuple[int, ...]
) -> tuple[jax.Array, ...]:
    """Points to test affineness at, scaled to the latents we are actually fitting.

    Each array is ``(n_data, latent_size)``, so it is ``n_data`` independent probe
    points rather than one -- the transform is applied per object. The second is an
    order of magnitude further out: a smooth nonlinearity's deviation from its tangent
    plane grows quadratically with amplitude, so it is ~100x more sensitive to a map
    that is only slightly non-affine. Fixed keys, so the verdict does not depend on
    when it is asked.
    """
    scale = 1.0 if latents is None else jnp.maximum(jnp.abs(latents).max(), 1.0)
    return (
        scale * jax.random.normal(jax.random.PRNGKey(0), shape),
        10 * scale * jax.random.normal(jax.random.PRNGKey(1), shape),
    )


def _linearize_outputs(
    model: LVM, data: PolluxData, current_params: dict[str, Any]
) -> dict[str, tuple[jax.Array, Any, Any]] | str:
    """Linearize every output that has data.

    Returns ``{output_name: (c, jvp, vjpT)}``, or a sentence naming the first output
    that cannot be used for a closed-form latent solve and saying why.
    """
    z0 = jnp.zeros((len(data), model.latent_size))
    probes = _latents_probe_points(current_params.get("latents"), z0.shape)

    linearized = {}
    for output_name in model.outputs:
        if output_name not in data:
            continue
        result = _linearize_latents(
            _output_predict_fn(model, output_name, current_params), z0, probes
        )
        if isinstance(result, str):
            return f"output '{output_name}' {result}"
        linearized[output_name] = result
    return linearized


def _participating_outputs(model: LVM, data: PolluxData) -> list[str]:
    """The registered outputs this dataset actually carries.

    A model is often applied to data holding only some of its outputs -- inferring
    labels from spectra alone, say. An output with no data contributes no likelihood
    term, so every part of the fit has to agree to leave it out: the closed-form
    solves, the SVI blocks, the prior initialization, the loss, and the default block
    list. Where they disagree, the ones that do not skip it fail looking for data or
    parameters that were never going to exist.
    """
    return [name for name in model.outputs if name in data]


def _has_learnable_params(
    transform: AbstractTransform, latent_size: int, data_size: int
) -> bool:
    """Whether a transform contributes any parameters to sample or solve for."""
    return bool(transform.get_expanded_priors(latent_size, data_size))


def _latents_from_data(model: LVM, data: PolluxData) -> jax.Array | None:
    """Observed latents, if some output reports them directly, else None.

    A model can observe its own latent vectors: the Cannon does, with labels behind a
    :class:`~pollux.models.transforms.NoOpTransform`. Where that is so, the observed
    values are a far better starting point than a draw from the prior -- the latents
    are then already at the answer, and the outputs that depend on them start from
    something meaningful rather than from noise.

    Whether an output is such a passthrough is *tested*, not assumed from its type: it
    must carry no learnable parameters, and must hand back a probe unchanged.
    """
    probe = jax.random.normal(jax.random.PRNGKey(0), (len(data), model.latent_size))

    for name, output in model.outputs.items():
        transform = output.data_transform
        if name not in data or _has_learnable_params(
            transform, model.latent_size, len(data)
        ):
            continue

        try:
            passthrough = transform.apply(probe)
        except (RuntimeError, TypeError, ValueError):
            # Takes parameters after all, or cannot accept latents of this shape
            continue

        if passthrough.shape == probe.shape and jnp.allclose(passthrough, probe):
            observed = data[name].data
            if jnp.all(jnp.isfinite(observed)):
                return observed

    return None


def _inverse_variance(
    model: LVM, data: PolluxData, output_name: str, params: dict[str, Any]
) -> jax.Array:
    """Inverse-variance weights for an output, as the *model* sees them.

    The err_transform is part of the model -- intrinsic scatter added in quadrature,
    a scale factor on reported uncertainties -- so the weights have to come through
    it rather than from the raw error column. Weighting by the raw errors while
    another block fits the scatter would leave the two blocks minimizing different
    objectives, and block coordinate descent only converges when they share one.

    Falls back to ones where no uncertainties were given at all.
    """
    output_data = data[output_name]
    err_pars = params.get(output_name, {}).get("err", {})
    err_transform = model.outputs[output_name].err_transform
    try:
        err = (
            err_transform.apply(output_data.err, **err_pars)
            if isinstance(err_pars, dict)
            else err_transform.apply(output_data.err, *err_pars)
        )
    except RuntimeError:
        # The err parameters are not available yet -- e.g. initial_params supplied
        # without them -- so the reported uncertainties are the best guess to hand
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
    model: LVM,
    data: PolluxData,
    current_params: dict[str, Any],
    latents_prior: dist.Distribution | None = None,
) -> jax.Array | str:
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
        The LVM instance.
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
        Optimal latent vectors of shape (n_stars, latent_size), or a sentence saying
        why no closed-form solve applies at these parameter values. It is a caller's
        job to fall back to SVI on that: whether the solve applies can change from
        cycle to cycle, because it is a property of the parameters as much as of the
        model.

    Notes
    -----
    Memory: no design matrix is ever formed. The largest temporaries are one output
    array, ``(n_data, output_size)``, and the accumulated ``(n_data, latent_size,
    latent_size)`` system -- the same footprint as the explicit-``A`` version this
    replaces, and independent of how the transform is composed.

    """
    n_data = len(data)
    latent_size = model.latent_size

    linearized = _linearize_outputs(model, data, current_params)
    if isinstance(linearized, str):
        return linearized

    # Sum the per-output contributions to the normal equations
    AtWA = jnp.zeros((n_data, latent_size, latent_size))
    AtWy = jnp.zeros((n_data, latent_size))

    for output_name, (c, jvp, vjpT) in linearized.items():
        output_data = data[output_name]
        w = _inverse_variance(model, data, output_name, current_params)

        # J^T W (y - c), one VJP
        AtWy = AtWy + vjpT(w * (output_data.data - c))[0]

        # J^T W J column by column: push a basis vector through J, weight it, pull it
        # back. Column k costs one JVP and one VJP, and no (n, output_size, latent_size)
        # intermediate is built.
        AtWA = AtWA + jnp.stack(
            [
                vjpT(w * jvp(jnp.broadcast_to(e, (n_data, latent_size))))[0]
                for e in jnp.eye(latent_size)
            ],
            axis=-1,
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
    model: LVM,
    data: PolluxData,
    output_name: str,
    params: dict[str, Any],
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
        The LVM instance.
    data
        The data to fit.
    output_name
        Name of the output to optimize.
    params
        Current parameter estimates. The latents supply the features, and the
        output's err_transform parameters supply the weights.

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
    latents = params["latents"]

    if output_name not in data:
        msg = f"No data found for output '{output_name}'"
        raise ValueError(msg)

    output_data = data[output_name]
    y = output_data.data  # (n_data, output_size)
    output_ivar = _inverse_variance(model, data, output_name, params)
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

    # One independent solve per output dimension -> (output_size, n_design)
    solution: jax.Array = jax.vmap(
        lambda y_dim, ivar_dim, rhs_row: weighted_least_squares(
            design, y_dim, ivar_dim, reg_matrix, rhs_row
        )
    )(y.T, output_ivar.T, rhs_extra)

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


def _string_to_parameter_block(model: LVM, name: str) -> ParameterBlock:
    """A block that asks for a closed form; :func:`_resolve_blocks` decides if it gets
    one, once there are parameters to test the transform with."""
    if name != "latents" and name.split(":", maxsplit=1)[0] not in model.outputs:
        msg = f"Unknown parameter block: '{name}'"
        raise ValueError(msg)
    # Error-transform parameters enter the likelihood through the variance, never as
    # least squares, so don't ask for a closed form we know cannot exist
    optimizer = None if name.endswith(":err") else "least_squares"
    return ParameterBlock(name=name, params=name, optimizer=optimizer)


def _least_squares_blocker(
    model: LVM,
    data: PolluxData,
    current_params: dict[str, Any],
    block: ParameterBlock,
) -> str | None:
    """Why this block cannot be solved in closed form, or None if it can be."""
    for spec in block.params_list:
        if spec == "latents":
            linearized = _linearize_outputs(model, data, current_params)
            if isinstance(linearized, str):
                return f"{linearized}, so the latents cannot be solved in closed form"
            continue

        output_name, _, param_type = spec.partition(":")
        if param_type == "err":
            return f"'{spec}' is an error transform, which has no closed-form solve"
        if _split_param_layer(model.outputs[output_name].data_transform) is None:
            return (
                f"output '{output_name}' does not end in a linear layer holding all "
                "of its parameters"
            )
    return None


def _resolve_blocks(
    model: LVM,
    data: PolluxData,
    current_params: dict[str, Any],
    blocks: list[ParameterBlock],
) -> list[ParameterBlock]:
    """Verify every block that wants a closed-form solve, downgrading those that can't.

    Blocks the caller explicitly assigned an SVI optimizer are left alone -- that was
    a choice, not a fallback, so it is not worth warning about.
    """
    resolved = []
    fallbacks = []
    for block in blocks:
        if block.optimizer == "least_squares":
            reason = _least_squares_blocker(model, data, current_params, block)
            if reason is not None:
                block = replace(block, optimizer=None)  # noqa: PLW2901
                fallbacks.append((block.name, reason))
        resolved.append(block)

    if fallbacks:
        detail = "\n".join(f"  {name:<12} - {reason}" for name, reason in fallbacks)
        warnings.warn(
            f"optimize_iterative could not use closed-form solves for "
            f"{len(fallbacks)} of {len(blocks)} blocks, falling back to SVI/Adam:\n"
            f"{detail}\n"
            'Silence with warnings.filterwarnings("ignore", '
            "category=pollux.exceptions.PolluxLinearizationWarning)",
            PolluxLinearizationWarning,
            stacklevel=3,
        )
    return resolved


def _build_initial_params_from_fixed(
    model: LVM,
    data: PolluxData,
    fixed_pars: dict[str, Any],
    blocks: list[ParameterBlock],
) -> dict[str, Any]:
    """Build initial params by merging fixed_pars with initialized optimized params."""
    initial: dict[str, Any] = dict(fixed_pars)

    if "latents" not in initial and any("latents" in b.params_list for b in blocks):
        observed = _latents_from_data(model, data)
        initial["latents"] = (
            jnp.zeros((len(data), model.latent_size)) if observed is None else observed
        )

    return initial


def optimize_iterative(
    model: LVM,
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
    parameter blocks, optimizing each while holding others fixed. Where a block's
    sub-problem is quadratic it is solved exactly by weighted least squares, which
    needs no learning rate and no step count.

    Which blocks those are is decided by measurement rather than by transform type,
    so it is a property of the model rather than a list of supported classes:

    - the **latents** can be solved exactly when every output with data is affine in
      them. That covers a bare linear map, but equally a slice of the latents feeding
      a linear branch, a ``ConcatenateTransform`` of linear children, or a linear map
      plus a fixed per-object offset.
    - an **output's own parameters** can be solved exactly when its transform ends in
      a linear layer that holds all of the transform's parameters. Anything before
      that layer is just run forward to make features -- which is what lets the
      Cannon's polynomial expansion work.

    Blocks that do not qualify fall back to SVI, and say so with a
    :class:`~pollux.exceptions.PolluxLinearizationWarning` naming each block and the
    reason. ``result.blocks`` reports what each block actually ran with. See the
    "Closed-form solves, found by linearization" page in the documentation for how
    the decision is made and what it does and does not guarantee.

    Initialization matters at least as much as any of that. Where an output reports
    the latents directly -- labels behind a
    :class:`~pollux.models.transforms.NoOpTransform`, as in the Cannon -- that
    output's data is used to start the latents instead of a draw from the prior, and
    the default block order flips so the outputs are fitted to those latents before
    the latents are touched. Otherwise the first latents step spends itself chasing
    prior-sampled output parameters. Pass ``initial_params`` to override.

    The default strategy alternates between:
    1. Optimize latents (with output parameters fixed)
    2. Optimize each output's parameters (with latents and other outputs fixed)

    Parameters
    ----------
    model
        The model to optimize.
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
    # Latents the data reports directly beat a draw from the prior, and also decide
    # which end of the model it makes sense to start from
    observed_latents = _latents_from_data(model, data)

    # Default blocks: the latents and every transform that has something to fit --
    # error transforms included, since their parameters are as much a part of the
    # model as the data transforms'. A transform carrying no learnable parameters
    # (NoOpTransform, a bare PolyFeatureTransform) gets no block.
    participating = _participating_outputs(model, data)

    if blocks is None:
        output_blocks = [
            f"{name}:{kind}"
            for name in participating
            for kind, transform in (
                ("data", model.outputs[name].data_transform),
                ("err", model.outputs[name].err_transform),
            )
            if _has_learnable_params(transform, model.latent_size, len(data))
        ]
        # Start from whichever end the data pins down. With the latents already at
        # their observed values, fit the outputs to them first: going the other way
        # would spend the first latents step chasing prior-sampled output parameters
        # and undo the head start.
        blocks = (
            [*output_blocks, "latents"]
            if observed_latents is not None
            else ["latents", *output_blocks]
        )

    # String specs are converted per element rather than by sniffing blocks[0], so a
    # mixed list works.
    _blocks: list[ParameterBlock] = [
        _string_to_parameter_block(model, b) if isinstance(b, str) else b
        for b in blocks
    ]

    # Build initial_params from fixed_pars if not provided
    if initial_params is None and fixed_pars is not None:
        initial_params = _build_initial_params_from_fixed(
            model, data, fixed_pars, _blocks
        )

    # Warn if any output has err_transform parameters that are neither being
    # optimized (in active blocks) nor intentionally held fixed (in fixed_pars)
    active_block_params = {b.params for b in _blocks}
    for output_name in participating:
        output = model.outputs[output_name]
        err_key = f"{output_name}:err"
        err_is_fixed = (
            fixed_pars is not None
            and output_name in fixed_pars
            and "err" in fixed_pars[output_name]
        )
        if (
            err_key not in active_block_params
            and not err_is_fixed
            and _has_learnable_params(
                output.err_transform, model.latent_size, len(data)
            )
        ):
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
        # names: the prior draw has to skip absent outputs too, or setup_numpyro
        # indexes the dataset for an output it does not hold
        predictive = Predictive(
            partial(model.default_numpyro_model, names=participating), num_samples=1
        )
        packed_samples = predictive(init_key, data)
        # Remove the batch dimension from num_samples=1, and filter out
        # observed samples (keys starting with "obs:")
        packed_samples = {
            k: v[0] for k, v in packed_samples.items() if not k.startswith("obs:")
        }
        current_params = model.unpack_numpyro_pars(packed_samples)

        if observed_latents is not None:
            current_params["latents"] = observed_latents
    else:
        current_params = initial_params

    # Now that there are parameters to probe with, settle which blocks get a
    # closed-form solve. This is not settled for good: whether a transform is affine
    # in the latents can depend on its parameters, and those move during the fit, so
    # the cycle loop downgrades a block if the closed form stops applying.
    _blocks = _resolve_blocks(model, data, current_params, _blocks)

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

        for index, block in enumerate(_blocks):
            if block.optimizer == "least_squares":
                outcome = _optimize_block_least_squares(
                    model, data, block, current_params, latents_prior
                )
                if not isinstance(outcome, str):
                    current_params = outcome
                    continue

                # Whether a closed form applies depends on the parameters, and they
                # have moved since the blocks were resolved -- a transform can be
                # affine in the latents at one set of values and not at another.
                # Downgrade for the rest of the fit rather than failing it.
                _blocks[index] = replace(block, optimizer=None)
                warnings.warn(
                    f"Block '{block.name}' can no longer use a closed-form solve "
                    f"({outcome}); it will use SVI/Adam for the rest of the fit.",
                    PolluxLinearizationWarning,
                    stacklevel=2,
                )
                block = _blocks[index]  # noqa: PLW2901

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
        blocks=_blocks,
    )


def _optimize_block_least_squares(
    model: LVM,
    data: PolluxData,
    block: ParameterBlock,
    current_params: dict[str, Any],
    latents_prior: dist.Distribution | None = None,
) -> dict[str, Any] | str:
    """Optimize a parameter block using least squares.

    Returns the updated parameters, or a sentence saying why the closed form no
    longer applies at these parameter values.
    """
    new_params = dict(current_params)

    for param_spec in block.params_list:
        if param_spec == "latents":
            solved = _solve_latents_least_squares(
                model, data, current_params, latents_prior
            )
            if isinstance(solved, str):
                return solved
            new_params["latents"] = solved
            continue

        # An output name, optionally qualified: only "data" params are solvable here
        output_name, _, param_type = param_spec.partition(":")
        if param_type not in ("", "data"):
            continue

        if output_name not in new_params:
            new_params[output_name] = {"data": {}, "err": {}}
        new_params[output_name]["data"] = _solve_output_params_least_squares(
            model, data, output_name, current_params
        )

    return new_params


def _optimize_block_numpyro(
    model: LVM,
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
        The LVM instance.
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

    partial_model = partial(
        model.default_numpyro_model,
        fixed_pars=packed_fixed_pars,
        latents_prior=latents_prior,
        names=_participating_outputs(model, data),
    )

    # Run SVI optimization, warm-started from where the last cycle left this block.
    # Without this the guide re-initializes from the prior every cycle, so the block
    # never accumulates progress and the overall loss is free to go *up* between
    # cycles -- which block coordinate descent must never do.
    svi_key, sample_key = jax.random.split(rng_key)
    guide = AutoDelta(
        partial_model,
        init_loc_fn=init_to_value(
            values=model.pack_numpyro_pars(current_params, ignore_missing=True)
        ),
    )
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
    model: LVM,
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
    model: LVM,
    data: PolluxData,
    params: dict[str, Any],
) -> float:
    """Compute the negative log likelihood loss."""
    latents = params["latents"]
    participating = _participating_outputs(model, data)
    # Predicting an absent output would demand parameters it was never fitted with,
    # and its prediction is discarded by the loop below in any case
    predictions = model.predict_outputs(params, latents, names=participating)

    total_loss = 0.0
    for output_name in participating:
        output_data = data[output_name]
        pred = predictions[output_name]
        obs = output_data.data

        # Gaussian negative log likelihood. The normalization is only constant
        # when the variance is; with an err_transform fitting the scatter, dropping
        # it would let the loss fall without bound as the modelled scatter grows.
        ivar = _inverse_variance(model, data, output_name, params)
        chi2 = (pred - obs) ** 2 * ivar
        total_loss = float(total_loss) + float(0.5 * jnp.sum(chi2 - jnp.log(ivar)))

    return total_loss
