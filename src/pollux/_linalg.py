"""Linear algebra shared by the closed-form fitters."""

__all__ = ["weighted_least_squares"]

import jax
import jax.numpy as jnp


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
