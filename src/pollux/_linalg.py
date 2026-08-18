"""Linear algebra stuff."""

__all__ = ("box_constrained_normal_equations", "nmf", "weighted_least_squares")

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr


def weighted_least_squares(
    design: jax.Array,
    y: jax.Array,
    ivar: jax.Array,
    reg_matrix: jax.Array,
    rhs_extra: jax.Array | float = 0.0,
) -> jax.Array:
    """Solve the regularized weighted normal equations for one response vector.

    Solves ``(D^T W D + reg) theta = D^T W y + rhs_extra``, where ``W`` is the
    diagonal matrix of inverse variances.

    Parameters
    ----------
    design
        Design matrix of shape ``(n_data, n_features)``.
    y
        Response vector of shape ``(n_data,)``.
    ivar
        Inverse variances of shape ``(n_data,)``.
    reg_matrix
        Regularization matrix of shape ``(n_features, n_features)``.
    rhs_extra
        Added to the right-hand side, e.g. the prior mean contribution
        ``alpha * mu``. Defaults to zero.

    Returns
    -------
    array
        Best-fit coefficients of shape ``(n_features,)``.
    """
    DtW = design.T * ivar  # (n_features, n_data)
    result: jax.Array = jnp.linalg.solve(DtW @ design + reg_matrix, DtW @ y + rhs_extra)
    return result


def box_constrained_normal_equations(
    H: jax.Array,
    b: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    n_sweeps: int = 100,
) -> jax.Array:
    """Minimize ``½ xᵀ H x - bᵀ x`` subject to ``lower <= x <= upper``.

    This is the constrained counterpart to :func:`weighted_least_squares`: the same
    normal equations, but with a box the solution has to stay inside. It is what a
    prior with bounded support -- a ``HalfNormal``, a ``TruncatedNormal`` -- turns the
    sub-problem into. Solving the unconstrained system and clipping afterwards is
    *not* equivalent and generally gives the wrong answer, because clipping one
    coordinate changes the optimum of the others.

    Uses cyclic coordinate descent. Each coordinate's update is the exact minimizer
    along that coordinate, clamped to its bounds::

        x_j <- clip((b_j - sum_{k != j} H_jk x_k) / H_jj, lower_j, upper_j)

    which needs no step size and decreases the objective monotonically. The systems
    here are small -- one per object or per output dimension, of size ``latent_size``
    -- so the coordinate loop is unrolled and the whole thing stays jittable with a
    fixed iteration count.

    Parameters
    ----------
    H
        Normal-equation matrices, shape ``(..., n, n)``. Assumed positive definite,
        which the prior's precision guarantees whenever it is nonzero.
    b
        Right-hand sides, shape ``(..., n)``.
    lower, upper
        Bounds, shape ``(n,)``. Infinities for unbounded sides.
    n_sweeps
        Number of coordinate sweeps.

    Returns
    -------
    array
        The constrained minimizer, shape ``(..., n)``.

    Examples
    --------
    The unconstrained minimum here is at ``(-1, 2)``, so a non-negativity constraint
    pins the first coordinate to zero:

    >>> import jax.numpy as jnp
    >>> from pollux._linalg import box_constrained_normal_equations
    >>> H = jnp.eye(2)
    >>> b = jnp.array([-1.0, 2.0])
    >>> lower, upper = jnp.zeros(2), jnp.full(2, jnp.inf)
    >>> box_constrained_normal_equations(H, b, lower, upper)
    Array([0., 2.], dtype=float32)
    """
    n = b.shape[-1]

    # Warm start from the unconstrained solution, clipped into the box: often already
    # optimal, and never worse than starting from the corner. A singular H gives
    # non-finite entries here, which fall back to zero.
    guess = jnp.linalg.solve(H, b[..., None])[..., 0]
    x0 = jnp.clip(jnp.where(jnp.isfinite(guess), guess, 0.0), lower, upper)

    def sweep(x: jax.Array, _: Any) -> tuple[jax.Array, None]:
        for j in range(n):
            # b_j - sum_{k != j} H_jk x_k, written as a full row product plus the
            # diagonal term back, so it stays one einsum rather than a masked gather
            off = (
                jnp.einsum("...k,...k->...", H[..., j, :], x) - H[..., j, j] * x[..., j]
            )
            x = x.at[..., j].set(
                jnp.clip((b[..., j] - off) / H[..., j, j], lower[j], upper[j])
            )
        return x, None

    x, _ = jax.lax.scan(sweep, x0, xs=None, length=n_sweeps)
    return x


def nmf(
    X: jax.Array, n_basis: int, key: jax.Array, n_iter: int = 128
) -> tuple[jax.Array, jax.Array]:
    """Factor a non-negative matrix as ``X ~ W H`` with both factors non-negative.

    This follows the Lee & Seung (2000) multiplicative update algorithm for non-negative
    matrix factorization (NMF). Each factor is rescaled by a ratio of non-negative
    quantities, so starting non-negative keeps it non-negative without a projection step
    or constrained optimizer.

    Parameters
    ----------
    X
        The matrix to factor, shape ``(n_data, output_size)``. Must be non-negative.
    n_basis
        Number of basis vectors, ``K``.
    key
        JAX random key, used for the (uniform, non-negative) starting factors.
    n_iter
        Number of multiplicative update steps.

    Returns
    -------
    tuple
        ``(W, H)`` with shapes ``(n_data, n_basis)`` and ``(n_basis, output_size)``.
    """
    eps = 1e-8
    n_data, output_size = X.shape
    w_key, h_key = jr.split(key)

    # start where W @ H already has roughly the right scale: with entries uniform on
    # [0, scale), (W @ H) averages n_basis * scale**2 / 4, so match that to mean(X)
    scale = 2 * jnp.sqrt(jnp.maximum(jnp.mean(X), eps) / n_basis)
    W = scale * jr.uniform(w_key, (n_data, n_basis))
    H = scale * jr.uniform(h_key, (n_basis, output_size))

    def step(
        carry: tuple[jax.Array, jax.Array], _: Any
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        W, H = carry
        W = W * (X @ H.T) / (W @ (H @ H.T) + eps)
        H = H * (W.T @ X) / ((W.T @ W) @ H + eps)
        return (W, H), None

    (W, H), _ = jax.lax.scan(step, (W, H), length=n_iter)
    return W, H
