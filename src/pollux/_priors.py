"""What a prior contributes to a closed-form solve, and whether it can contribute.

The closed-form (least squares) solves in :mod:`pollux.models.iterative` work because
the objective is quadratic in the block being solved. The likelihood supplies one
quadratic; the prior has to supply another, or it cannot be folded into the normal
equations at all. This module decides which priors qualify, and what they contribute.

Getting this wrong is not a performance problem, it is a correctness problem: a prior
that is silently dropped produces a fit that violates it, and the result looks
plausible. So the rule here is that anything not explicitly recognized is refused, and
the caller falls back to an optimizer that can honor it.
"""

__all__ = ("PriorTerm", "prior_term", "support_bounds")

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

#: numpyro's ``TruncatedNormal`` is a factory function, not a class: it returns one of
#: these three depending on which bounds were supplied, so ``isinstance`` has to name
#: them explicitly. All three expose the untruncated ``Normal`` as ``.base_dist``.
_TRUNCATED = (
    dist.LeftTruncatedDistribution,
    dist.RightTruncatedDistribution,
    dist.TwoSidedTruncatedDistribution,
)


@dataclass(frozen=True)
class PriorTerm:
    """The quadratic and the box that a prior contributes to a solve.

    The prior's negative log density is ``½ precision (x - mean)²`` up to a constant,
    restricted to ``lower <= x <= upper``. An unbounded side is an infinity, in which
    case the solve does not need a constrained routine.

    Parameters
    ----------
    precision
        The quadratic coefficient, ``1 / scale**2`` for a Gaussian. Zero for a flat
        prior. May be an array, for a prior that differs element by element.
    mean
        The value the prior pulls toward.
    lower, upper
        Bounds implied by the prior's support.
    """

    precision: jax.Array | float
    mean: jax.Array | float
    lower: jax.Array | float
    upper: jax.Array | float
    event_shape: tuple[int, ...] = ()

    @property
    def correlated(self) -> bool:
        """Whether ``precision`` is a matrix coupling one axis of the parameter.

        Cannot be inferred from ``precision.ndim``: an elementwise prior on a matrix
        parameter has a two-dimensional precision too, it just is not a matrix in the
        linear-algebra sense.
        """
        return bool(self.event_shape)

    @property
    def bounded(self) -> bool:
        """Whether either bound is finite, so a constrained solve is needed."""
        return bool(jnp.any(jnp.isfinite(jnp.asarray(self.lower)))) or bool(
            jnp.any(jnp.isfinite(jnp.asarray(self.upper)))
        )


def support_bounds(support: Any) -> tuple[jax.Array | float, jax.Array | float]:
    """Bounds implied by a numpyro constraint, as ``(lower, upper)``.

    Unbounded sides come back as infinities. Constraints that wrap another one --
    the independent constraint an ``ImproperUniform`` carries, for instance -- are
    unwrapped first.

    The bounds have to be read off the constraint rather than inferred from where
    ``log_prob`` returns ``-inf``: numpyro does not mask out-of-support values unless
    the distribution was built with ``validate_args=True``, so
    ``HalfNormal(1.0).log_prob(-1.7)`` is a perfectly finite number.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> from pollux._priors import support_bounds
    >>> support_bounds(dist.Normal(0.0, 1.0).support)
    (-inf, inf)
    >>> support_bounds(dist.HalfNormal(1.0).support)
    (0.0, inf)
    """
    base = getattr(support, "base_constraint", None)
    if base is not None:
        support = base

    lower = getattr(support, "lower_bound", None)
    upper = getattr(support, "upper_bound", None)
    return (
        -jnp.inf if lower is None else lower,
        jnp.inf if upper is None else upper,
    )


def prior_term(prior: dist.Distribution) -> PriorTerm | None:
    """What ``prior`` contributes to a normal-equation solve, or None if it cannot.

    Returns None for any prior whose negative log density is not quadratic -- a
    ``Laplace`` or a ``StudentT``, say -- because such a prior cannot be expressed as
    a precision and a mean, and quietly approximating it by one would silently change
    the model. The caller is expected to fall back to an optimizer that can handle it.

    A ``MultivariateNormal`` is quadratic too, and comes back with a precision
    *matrix* and the event shape it correlates. Whether that axis is one the solve can
    use is for the caller to decide: correlating the latent axis replaces the ridge,
    whereas correlating an output axis would couple what are otherwise independent
    per-output-dimension solves into a single system, which no closed-form path here
    can do. See :func:`~pollux.models.iterative.optimize_iterative`, which refuses the
    latter and falls back to SVI.

    Parameters
    ----------
    prior
        A numpyro distribution.

    Returns
    -------
    PriorTerm or None
        The prior's contribution, or None when it has none that a linear solve can use.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> from pollux._priors import prior_term
    >>> term = prior_term(dist.Normal(2.0, 0.5))
    >>> float(term.precision), float(term.mean), term.bounded
    (4.0, 2.0, False)

    A bounded prior reports its box, and needs a constrained solve:

    >>> term = prior_term(dist.HalfNormal(1.0))
    >>> float(term.precision), float(term.lower), term.bounded
    (1.0, 0.0, True)

    Anything that is not a bounded quadratic is refused outright:

    >>> prior_term(dist.Laplace(0.0, 1.0)) is None
    True
    """
    # An expanded prior wraps the real one and does not forward its parameters
    while isinstance(prior, dist.ExpandedDistribution):
        prior = prior.base_dist

    lower, upper = support_bounds(prior.support)

    def gaussian(scale: Any, loc: Any) -> PriorTerm:
        # numpyro ships no py.typed marker, so the types of a distribution's own
        # parameters come back loose; pin them to arrays on the way in
        return PriorTerm(1.0 / jnp.asarray(scale) ** 2, jnp.asarray(loc), lower, upper)

    if isinstance(prior, dist.Normal):
        return gaussian(prior.scale, prior.loc)

    if isinstance(prior, dist.HalfNormal):
        return gaussian(prior.scale, 0.0)

    if isinstance(prior, _TRUNCATED):
        base = prior.base_dist
        if not isinstance(base, dist.Normal):
            return None
        return gaussian(base.scale, base.loc)

    # Quadratic, but coupling a whole axis rather than acting element by element.
    # Which axis of the parameter that is depends on where the prior is used, so the
    # event shape is reported and the caller matches it up.
    if isinstance(prior, dist.MultivariateNormal):
        return PriorTerm(
            jnp.asarray(prior.precision_matrix),
            jnp.asarray(prior.loc),
            lower,
            upper,
            event_shape=tuple(prior.event_shape),
        )

    # Flat priors contribute no quadratic term at all, only (possibly) a box
    if isinstance(prior, (dist.Uniform, dist.ImproperUniform)):
        return PriorTerm(0.0, 0.0, lower, upper)

    return None
