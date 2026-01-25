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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import Predictive
from tqdm.auto import tqdm

from ..data import PolluxData
from .transforms import (
    AffineTransform,
    LinearTransform,
    OffsetTransform,
    TransformSequence,
)

if TYPE_CHECKING:
    from .lux import Lux


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
        If None, uses numpyro SVI with the default optimizer.
    optimizer_kwargs
        Keyword arguments to pass to the optimizer constructor (e.g., ``step_size`` for
        Adam). Ignored when using ``"least_squares"`` optimizer.
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
    history
        Optional detailed history of losses per block per cycle.

    """

    params: dict[str, Any]
    losses_per_cycle: list[float]
    n_cycles: int
    converged: bool
    history: list[dict[str, Any]] | None = None


def _is_linear_transform(transform: Any) -> bool:
    """Check if a transform is linear (amenable to least squares).

    Note: TransformSequence is not supported for iterative optimization,
    even if all component transforms are linear.
    """
    return bool(
        isinstance(transform, (LinearTransform, AffineTransform, OffsetTransform))
    )


def _get_regularization_from_prior(
    prior: dist.Distribution,
    fallback: float = 1e-6,
) -> tuple[jax.Array | float, jax.Array | float]:
    """Extract regularization parameters from a prior distribution.

    Parameters
    ----------
    prior
        A numpyro distribution. For Normal distributions, extracts the precision.
        For other distributions, returns the fallback regularization.
    fallback
        Default regularization strength if the prior is not Normal.

    Returns
    -------
    regularization
        The regularization strength alpha = 1 / scale**2.
    prior_mean
        The prior mean μ (for regularization toward non-zero mean).

    Notes
    -----
    Currently only supports Normal distributions. For other priors,
    uses the fallback regularization with zero mean.
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
    return fallback, 0.0


def _solve_latents_least_squares(
    model: Lux,
    data: PolluxData,
    current_params: dict[str, Any],
    latents_prior: dist.Distribution | None = None,
) -> jax.Array:
    """Solve for optimal latents using weighted least squares.

    For linear models: y = A @ z, we solve for z using the normal equations:
        z = (A^T W A + λI)^{-1} A^T W y

    When there are multiple outputs, we combine them into a block-diagonal system:
        [y1]   [A1  0 ]       [A1^T W1 A1 + ... ] z = [A1^T W1 y1 + ...]
        [y2] = [0  A2 ] z  →
        ...

    This sums the contributions from each output to form a single linear system.

    Parameters
    ----------
    model
        The Lux instance.
    data
        The data to fit.
    current_params
        Current parameter estimates (used for A matrices).
    latents_prior
        Prior distribution for latents. If None, uses Normal(0, 1).
        The regularization strength is extracted from this prior.

    Returns
    -------
    latents
        Optimal latent vectors of shape (n_stars, latent_size).

    Notes
    -----
    Memory considerations: Rather than forming a large block-diagonal matrix,
    we accumulate the contributions to the normal equations from each output.
    This is O(n_latents^2) memory rather than O(n_outputs * n_output_size * n_latents).

    """
    n_data = len(data)
    latent_size = model.latent_size

    # Initialize normal equations: sum over outputs
    # AtWA shape: (n_data, latent_size, latent_size)
    # AtWy shape: (n_data, latent_size)
    AtWA = jnp.zeros((n_data, latent_size, latent_size))
    AtWy = jnp.zeros((n_data, latent_size))

    for output_name, lux_output in model.outputs.items():
        transform = lux_output.data_transform
        if isinstance(transform, TransformSequence):
            msg = (
                f"Output '{output_name}' uses a TransformSequence; "
                "iterative least squares only supports single transforms"
            )
            raise ValueError(msg)
        if not _is_linear_transform(transform):
            msg = f"Output '{output_name}' has a non-linear transform; cannot use least squares"
            raise ValueError(msg)

        if output_name not in data:
            continue

        output_data = data[output_name]
        y = output_data.data  # (n_data, output_size)
        err = output_data.err

        # Compute inverse variance weights
        w = 1.0 / err**2 if err is not None and jnp.any(err > 0) else jnp.ones_like(y)

        # Get the transformation matrix A from current params
        # For LinearTransform: y = A @ z, so A has shape (output_size, latent_size)
        output_params = current_params.get(output_name, {}).get("data", {})
        A = output_params.get("A")

        if A is None:
            msg = f"Could not find matrix 'A' for output '{output_name}'"
            raise ValueError(msg)

        # Accumulate contributions to normal equations
        # A: (output_size, latent_size)
        # y: (n_data, output_size)
        # w: (n_data, output_size) inverse variances

        # For each data point i:
        #   AtWA[i] += A.T @ diag(w[i]) @ A
        #   AtWy[i] += A.T @ (w[i] * y[i])

        # AtWA[i] = A.T @ diag(w[i]) @ A = sum_j w[i,j] * A[j,:].T @ A[j,:]
        AtWA = AtWA + jnp.einsum("nj,jk,jl->nkl", w, A, A)

        # A.T @ (w * y) for each data point
        # (w * y): (n_data, output_size)
        AtWy = AtWy + jnp.einsum("nj,jk,nj->nk", w, A, y)

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
) -> dict[str, Any]:
    """Solve for optimal output parameters using weighted least squares.

    For linear model y = A @ z, solving for A (treating z as fixed):
        vec(A) = (Z ⊗ I)^{-1} vec(Y)

    In practice, we solve per-pixel: for each output dimension j,
        A[j, :] = (Z^T W_j Z + λI)^{-1} Z^T W_j y[:, j]

    where Z is the latents matrix and W_j = diag(1/err[:, j]^2).

    The regularization strength λ is extracted from the transform's prior on A.

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
        Optimized parameters for this output in nested format.

    """
    transform = model.outputs[output_name].data_transform
    if isinstance(transform, TransformSequence):
        msg = (
            f"Output '{output_name}' uses a TransformSequence; "
            "iterative least squares only supports single transforms"
        )
        raise ValueError(msg)
    if not _is_linear_transform(transform):
        msg = f"Output '{output_name}' has a non-linear transform; cannot use least squares"
        raise ValueError(msg)

    if output_name not in data:
        msg = f"No data found for output '{output_name}'"
        raise ValueError(msg)

    output_data = data[output_name]
    y = output_data.data  # (n_data, output_size)
    err = output_data.err
    # Compute inverse variance weights
    if err is not None and jnp.any(err > 0):
        output_ivar = 1.0 / (err**2)
    else:
        output_ivar = jnp.ones_like(y)

    latent_size = model.latent_size
    output_size = y.shape[1]

    # Get regularization from the transform's prior on A
    a_prior = transform.priors.get("A", dist.Normal(0.0, 1.0))

    reg_strength, prior_mean = _get_regularization_from_prior(a_prior)
    reg_matrix = reg_strength * jnp.eye(latent_size)

    # Prior mean for A - typically 0, but could be non-zero
    # prior_mean could be scalar or array; handle both
    if hasattr(prior_mean, "shape") and prior_mean.shape:
        # Array prior mean - need to extract row for each output dim
        # For now, assume scalar or broadcast
        prior_mean_contrib = reg_strength * jnp.broadcast_to(
            prior_mean, (output_size, latent_size)
        )
    else:
        prior_mean_contrib = reg_strength * jnp.full(
            (output_size, latent_size), prior_mean
        )

    def fit_single_output_dim(
        dim_data: tuple[jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        """Fit parameters for a single output dimension."""
        y_dim, w_dim, prior_mean_row = dim_data  # (n_data,), (n_data,), (latent_size,)

        # Weighted normal equations: (Z^T W Z + λI) @ a = Z^T W y + λ μ
        ZtW = latents.T * w_dim  # (latent_size, n_data)
        ZtWZ = ZtW @ latents  # (latent_size, latent_size)
        ZtWy = ZtW @ y_dim  # (latent_size,)

        # Solve with regularization and prior mean
        result: jax.Array = jnp.linalg.solve(ZtWZ + reg_matrix, ZtWy + prior_mean_row)
        return result

    # Vectorize over output dimensions
    dim_data = (y.T, output_ivar.T, prior_mean_contrib)  # (output_size, ...) each
    A: jax.Array = jax.vmap(fit_single_output_dim)(
        dim_data
    )  # (output_size, latent_size)

    return {"A": A}


def optimize_iterative(
    model: Lux,
    data: PolluxData,
    blocks: list[ParameterBlock] | None = None,
    max_cycles: int = 10,
    tol: float = 1e-4,
    rng_key: jax.Array | None = None,
    initial_params: dict[str, Any] | None = None,
    latents_prior: dist.Distribution | None = None,
    progress: bool = True,
    record_history: bool = False,
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
        List of ParameterBlock specifications. If None, uses a default strategy
        that alternates between latents and each output.
    max_cycles
        Maximum number of full optimization cycles.
    tol
        Convergence tolerance. Stops when relative change in loss < tol.
    rng_key
        Random key for SVI-based optimization. Required if any block uses SVI.
    initial_params
        Initial parameter values. If None, initialized from priors.
    latents_prior
        Prior distribution for latents. If None, uses Normal(0, 1).
        Used to determine regularization strength for latent least squares.
    progress
        Whether to display a tqdm progress bar showing optimization progress.
    record_history
        Whether to record detailed per-block loss history.

    Returns
    -------
    IterativeOptimizationResult
        The optimization result containing optimized parameters and convergence info.

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

    """
    # Default blocks: alternate between latents and each output
    if blocks is None:
        blocks = [
            ParameterBlock(
                name="latents",
                params="latents",
                optimizer="least_squares" if _all_outputs_linear(model) else None,
            )
        ]
        for output_name, lux_output in model.outputs.items():
            transform = lux_output.data_transform
            blocks.append(
                ParameterBlock(
                    name=output_name,
                    params=f"{output_name}:data",
                    optimizer="least_squares"
                    if _is_linear_transform(transform)
                    else None,
                )
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
    history: list[dict[str, Any]] = []

    prev_loss = float("inf")

    # Set up progress bar
    pbar = tqdm(
        range(max_cycles),
        desc="Iterative optimization",
        disable=not progress,
    )

    for cycle in pbar:
        cycle_history: dict[str, Any] = {}

        for block in blocks:
            if block.optimizer == "least_squares":
                current_params = _optimize_block_least_squares(
                    model, data, block, current_params, latents_prior
                )
            else:
                # if rng_key is None:
                #     msg = "rng_key required for SVI-based optimization"
                #     raise ValueError(msg)
                # rng_key, subkey = jax.random.split(rng_key)
                # current_params = _optimize_block_svi(
                #     model, data, block, current_params, subkey
                # )
                msg = "non-least-squares block optimization not yet implemented"
                raise NotImplementedError(msg)

            if record_history:
                # Could compute loss here per block if needed
                cycle_history[block.name] = None

        # Compute loss at end of cycle
        loss = _compute_loss(model, data, current_params)
        losses_per_cycle.append(float(loss))

        if record_history:
            history.append(cycle_history)

        # Update progress bar with loss info
        rel_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-8)
        pbar.set_postfix(
            loss=f"{loss:.4g}",
            rel_change=f"{rel_change:.2e}",
        )

        # Check convergence
        if rel_change < tol:
            # Update to show successful convergence (green/complete state)
            pbar.set_description("Converged")
            pbar.update(max_cycles - pbar.n)  # Complete the bar
            pbar.set_postfix(loss=f"{loss:.4g}")
            pbar.close()
            return IterativeOptimizationResult(
                params=current_params,
                losses_per_cycle=losses_per_cycle,
                n_cycles=cycle + 1,
                converged=True,
                history=history,
            )
        prev_loss = loss

    pbar.close()

    return IterativeOptimizationResult(
        params=current_params,
        losses_per_cycle=losses_per_cycle,
        n_cycles=max_cycles,
        converged=False,
        history=history,
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
    params = block.params
    if isinstance(params, str):
        params = [params]

    new_params = dict(current_params)

    for param_spec in params:
        if param_spec == "latents":
            new_latents = _solve_latents_least_squares(
                model, data, current_params, latents_prior
            )
            new_params["latents"] = new_latents

        elif ":" in param_spec:
            output_name, param_type = param_spec.split(":", 1)
            if param_type == "data":
                output_params = _solve_output_params_least_squares(
                    model,
                    data,
                    output_name,
                    current_params["latents"],
                )
                if output_name not in new_params:
                    new_params[output_name] = {"data": {}, "err": {}}
                new_params[output_name]["data"] = output_params
        else:
            # Just output name - optimize data params
            output_params = _solve_output_params_least_squares(
                model, data, param_spec, current_params["latents"]
            )
            if param_spec not in new_params:
                new_params[param_spec] = {"data": {}, "err": {}}
            new_params[param_spec]["data"] = output_params

    return new_params


# TODO: could also have an _optimize_block_adam for non-linear outputs


def _build_fixed_pars(
    model: Lux,
    current_params: dict[str, Any],
    optimize_params: list[str],
) -> dict[str, Any]:
    """Build fixed_pars dict containing everything not being optimized."""
    fixed = {}

    # Check if latents should be fixed
    if "latents" not in optimize_params:
        fixed["latents"] = current_params.get("latents")

    # Check each output
    for output_name in model.outputs:
        data_fixed = (
            f"{output_name}:data" not in optimize_params
            and output_name not in optimize_params
        )
        err_fixed = (
            f"{output_name}:err" not in optimize_params
            and output_name not in optimize_params
        )

        if data_fixed or err_fixed:
            output_params = current_params.get(output_name, {})
            if output_name not in fixed:
                fixed[output_name] = {}
            output_fixed = dict(fixed[output_name])  # type: ignore[arg-type]
            if data_fixed and "data" in output_params:
                output_fixed["data"] = output_params["data"]
            if err_fixed and "err" in output_params:
                output_fixed["err"] = output_params["err"]

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
        err = output_data.err

        if err is None or not jnp.any(err > 0):
            err = jnp.ones_like(obs)

        # Gaussian negative log likelihood (ignoring constant)
        residuals = (pred - obs) / err
        total_loss = float(total_loss) + float(0.5 * jnp.sum(residuals**2))

    return total_loss
