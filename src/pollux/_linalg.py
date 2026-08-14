"""Linear algebra stuff."""

__all__ = ("nmf", "weighted_least_squares")

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
