"""Tests for classifying priors by what they contribute to a closed-form solve."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pollux._linalg import box_constrained_normal_equations
from pollux._priors import PriorTerm, prior_term, support_bounds

jax.config.update("jax_enable_x64", True)


class TestSupportBounds:
    """Bounds come off the constraint, not off where log_prob goes to -inf."""

    @pytest.mark.parametrize(
        ("prior", "expected"),
        [
            (dist.Normal(0.0, 1.0), (-jnp.inf, jnp.inf)),
            (dist.HalfNormal(1.0), (0.0, jnp.inf)),
            (dist.TruncatedNormal(scale=1.0, high=0.0), (-jnp.inf, 0.0)),
            (dist.TruncatedNormal(0.0, 1.0, low=-2.0, high=3.0), (-2.0, 3.0)),
            (dist.Uniform(-5.0, 5.0), (-5.0, 5.0)),
        ],
    )
    def test_bounds(self, prior, expected):
        lower, upper = support_bounds(prior.support)
        assert jnp.isclose(lower, expected[0])
        assert jnp.isclose(upper, expected[1])

    def test_improper_uniform_unwraps_its_independent_constraint(self):
        prior = dist.ImproperUniform(dist.constraints.real, (), ())
        assert support_bounds(prior.support) == (-jnp.inf, jnp.inf)

    def test_log_prob_is_not_a_support_test(self):
        """The trap this module exists to avoid.

        numpyro does not mask out-of-support values unless validate_args is set, so a
        finite log_prob says nothing about the support.
        """
        assert jnp.isfinite(dist.HalfNormal(1.0).log_prob(jnp.array(-1.7)))
        assert support_bounds(dist.HalfNormal(1.0).support)[0] == 0.0


class TestPriorTerm:
    """What each prior contributes, and which ones are refused."""

    def test_normal(self):
        """These four cases are the contract the previous implementation had."""
        assert jnp.isclose(prior_term(dist.Normal(0.0, 1.0)).precision, 1.0)
        assert jnp.isclose(prior_term(dist.Normal(0.0, 1.0)).mean, 0.0)
        assert jnp.isclose(prior_term(dist.Normal(0.0, 0.5)).precision, 4.0)
        assert jnp.isclose(prior_term(dist.Normal(1.0, 2.0)).precision, 0.25)
        assert jnp.isclose(prior_term(dist.Normal(1.0, 2.0)).mean, 1.0)

    def test_improper_uniform(self):
        term = prior_term(dist.ImproperUniform(dist.constraints.real, (), ()))
        assert jnp.isclose(term.precision, 0.0)
        assert jnp.isclose(term.mean, 0.0)
        assert not term.bounded

    @pytest.mark.parametrize(
        ("prior", "precision", "mean"),
        [
            (dist.HalfNormal(1.0), 1.0, 0.0),
            (dist.HalfNormal(0.5), 4.0, 0.0),
            (dist.TruncatedNormal(scale=1.0, high=0.0), 1.0, 0.0),
            (dist.TruncatedNormal(2.0, 0.5, low=0.0), 4.0, 2.0),
            (dist.Uniform(-1.0, 1.0), 0.0, 0.0),
        ],
    )
    def test_bounded_quadratics(self, prior, precision, mean):
        term = prior_term(prior)
        assert term is not None
        assert jnp.isclose(term.precision, precision)
        assert jnp.isclose(term.mean, mean)
        assert term.bounded

    @pytest.mark.parametrize(
        "prior",
        [
            dist.Laplace(0.0, 1.0),
            dist.StudentT(3.0),
            dist.Cauchy(0.0, 1.0),
            dist.LogNormal(0.0, 1.0),
            dist.Gamma(2.0, 1.0),
            dist.Exponential(1.0),
        ],
    )
    def test_refuses_non_quadratics(self, prior):
        """Anything that is not a bounded quadratic is refused rather than approximated."""
        assert prior_term(prior) is None

    def test_unwraps_expanded(self):
        """An expanded prior hides its parameters behind a wrapper."""
        term = prior_term(dist.Normal(1.0, 2.0).expand((4,)))
        assert term is not None
        assert jnp.isclose(term.precision, 0.25)
        assert jnp.isclose(term.mean, 1.0)

    def test_array_valued_parameters_survive(self):
        term = prior_term(dist.Normal(jnp.zeros(4), jnp.full(4, 0.5)))
        assert term.precision.shape == (4,)
        assert jnp.allclose(term.precision, 4.0)

    def test_unbounded_is_not_bounded(self):
        assert not PriorTerm(1.0, 0.0, -jnp.inf, jnp.inf).bounded
        assert PriorTerm(1.0, 0.0, 0.0, jnp.inf).bounded
        assert PriorTerm(1.0, 0.0, -jnp.inf, 3.0).bounded


class TestBoxConstrainedSolve:
    """The constrained counterpart to weighted_least_squares."""

    @pytest.fixture
    def system(self):
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (6, 4))
        H = M.T @ M + jnp.eye(4)  # positive definite
        b = jax.random.normal(jax.random.PRNGKey(1), (4,))
        return H, b

    def test_matches_linear_solve_when_unbounded(self, system):
        H, b = system
        lower, upper = jnp.full(4, -jnp.inf), jnp.full(4, jnp.inf)
        got = box_constrained_normal_equations(H, b, lower, upper)
        assert jnp.allclose(got, jnp.linalg.solve(H, b), atol=1e-8)

    def test_satisfies_the_bounds(self, system):
        H, b = system
        lower, upper = jnp.zeros(4), jnp.full(4, jnp.inf)
        x = box_constrained_normal_equations(H, b, lower, upper)
        assert jnp.all(x >= 0)

    def test_kkt_conditions(self, system):
        """The real correctness check, needing no reference implementation.

        At the constrained optimum every coordinate is either interior with zero
        gradient, or pinned at a bound with the gradient pushing it further out.
        """
        H, b = system
        lower, upper = jnp.zeros(4), jnp.full(4, 0.3)
        x = box_constrained_normal_equations(H, b, lower, upper)
        grad = H @ x - b  # gradient of the objective

        at_lower = jnp.isclose(x, lower, atol=1e-8)
        at_upper = jnp.isclose(x, upper, atol=1e-8)
        interior = ~(at_lower | at_upper)

        assert jnp.all(jnp.abs(grad[interior]) < 1e-6)  # stationary where free
        assert jnp.all(grad[at_lower] >= -1e-6)  # pushing down, held at the floor
        assert jnp.all(grad[at_upper] <= 1e-6)  # pushing up, held at the ceiling

    def test_clipping_afterwards_is_not_equivalent(self, system):
        """Why this routine has to exist at all."""
        H, b = system
        lower, upper = jnp.zeros(4), jnp.full(4, jnp.inf)
        proper = box_constrained_normal_equations(H, b, lower, upper)
        naive = jnp.clip(jnp.linalg.solve(H, b), lower, upper)

        def objective(x):
            return 0.5 * x @ H @ x - b @ x

        assert not jnp.allclose(proper, naive)
        assert objective(proper) < objective(naive)

    def test_batched(self, system):
        H, b = system
        Hs = jnp.stack([H, 2 * H])
        bs = jnp.stack([b, -b])
        lower, upper = jnp.zeros(4), jnp.full(4, jnp.inf)
        x = box_constrained_normal_equations(Hs, bs, lower, upper)
        assert x.shape == (2, 4)
        assert jnp.all(x >= 0)
        # each row must solve its own system
        for i in range(2):
            single = box_constrained_normal_equations(Hs[i], bs[i], lower, upper)
            assert jnp.allclose(x[i], single, atol=1e-8)


class TestCorrelatedPriors:
    """MultivariateNormal is quadratic, but couples a whole axis."""

    def test_reports_a_precision_matrix_and_its_event_shape(self):
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        term = prior_term(dist.MultivariateNormal(jnp.array([1.0, -1.0]), cov))
        assert term is not None
        assert term.correlated
        assert term.event_shape == (2,)
        assert jnp.allclose(term.precision, jnp.linalg.inv(cov))
        assert jnp.allclose(term.mean, jnp.array([1.0, -1.0]))
        assert not term.bounded

    def test_elementwise_priors_are_not_correlated(self):
        """Even when their precision happens to be two-dimensional."""
        term = prior_term(dist.Normal(jnp.zeros((3, 2)), jnp.ones((3, 2))))
        assert not term.correlated
        assert term.precision.shape == (3, 2)

    def test_diagonal_covariance_matches_the_equivalent_normal(self):
        mvn = prior_term(
            dist.MultivariateNormal(jnp.zeros(3), covariance_matrix=0.25 * jnp.eye(3))
        )
        normal = prior_term(dist.Normal(0.0, 0.5))
        assert jnp.allclose(jnp.diag(mvn.precision), normal.precision)
        assert jnp.allclose(mvn.precision - jnp.diag(jnp.diag(mvn.precision)), 0.0)
