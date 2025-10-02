import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux.models.transforms import (
    FunctionTransform,
    LinearTransform,
    OffsetTransform,
    TransformSequence,
)


def test_linear_transform():
    n_stars = 64
    n_latents = 16
    n_out = 8
    rng = np.random.default_rng(42)

    trans = LinearTransform(output_size=n_out)
    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = rng.random((n_out, n_latents))

    # Test direct computation
    expected = np.array([A @ latents[i] for i in range(n_stars)])
    result = trans.apply(latents, A=A)
    assert np.allclose(result, expected)

    # Test with prior
    trans_prior = LinearTransform(
        output_size=n_out, param_priors={"A": dist.Normal(0.0, 1.0)}
    )
    result_prior = trans_prior.apply(latents, A=A)
    assert np.allclose(result_prior, expected)


def test_offset_transform():
    n_stars = 64
    n_dim = 8
    rng = np.random.default_rng(42)

    trans = OffsetTransform(output_size=n_stars, vmap=False)
    x = jnp.array(rng.random((n_stars, n_dim)))
    b = jnp.array(rng.random((n_stars, n_dim)))

    # Test direct computation
    expected = x + b
    result = trans.apply(x, b=b)
    assert np.allclose(result, expected)

    # Test with prior
    trans_prior = OffsetTransform(
        output_size=n_stars, vmap=False, param_priors={"b": dist.Normal(0.0, 1.0)}
    )
    result_prior = trans_prior.apply(x, b=b)
    assert np.allclose(result_prior, expected)


def test_transform_sequence():
    n_stars = 128
    n_latents = 32
    n_out = 8
    rng = np.random.default_rng(0)

    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=8),
            OffsetTransform(output_size=n_stars, vmap=False),
        )
    )

    latents = rng.random((n_stars, n_latents))

    A = rng.random((n_out, n_latents))
    b = rng.random((n_stars, n_out))

    tmp = np.array([A @ latents[i] for i in range(n_stars)])
    tmp += b

    # Test new parameter format with *args (list of dicts)
    tmp2 = trans.apply(jnp.array(latents), {"A": A}, {"b": b})
    assert np.allclose(tmp2, tmp)

    # Test new flat parameter format with "{index}:{param}" naming
    tmp3 = trans.apply(jnp.array(latents), **{"0:A": A, "1:b": b})
    assert np.allclose(tmp3, tmp)


def test_transform_sequence_priors():
    n_stars = 128
    n_latents = 32
    n_out = 8
    rng = np.random.default_rng(0)

    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=8, param_priors={"A": dist.Laplace()}),
            OffsetTransform(
                output_size=n_stars,
                vmap=False,
                param_priors={"b": dist.Normal(11.0, 3.0)},
            ),
        )
    )

    latents = jnp.array(rng.random((n_stars, n_latents)))

    A = rng.random((n_out, n_latents))
    b = rng.random((n_stars, n_out))

    tmp = np.array([A @ latents[i] for i in range(n_stars)])
    tmp += b

    # Test new parameter format with list of dicts
    params = {"mags": [{"A": A}, {"b": b}]}

    model = plx.LuxModel(latent_size=n_latents)
    model.register_output("mags", trans)
    out = model.predict_outputs(latents, params)

    assert np.allclose(out["mags"], tmp)

    # Test new flat parameter format
    params_flat = {"mags": {"0:A": A, "1:b": b}}
    out_flat = model.predict_outputs(latents, params_flat)
    assert np.allclose(out_flat["mags"], tmp)


def test_transform_sequence_new_parameter_structure():
    """Test the new tuple-based parameter structure in TransformSequence."""
    n_stars = 64
    n_out = 8

    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            OffsetTransform(output_size=n_stars, vmap=False),
        )
    )

    # Test that param_priors and param_shapes are properly stored as tuples
    assert len(trans.param_priors) == 2
    assert len(trans.param_shapes) == 2

    # First transform (LinearTransform) should have 'A' parameter
    assert "A" in trans.param_priors[0]
    assert "A" in trans.param_shapes[0]

    # Second transform (OffsetTransform) should have 'b' parameter
    assert "b" in trans.param_priors[1]
    assert "b" in trans.param_shapes[1]


def test_transform_sequence_parameter_validation():
    """Test parameter validation in the new apply method."""
    n_stars = 16
    n_latents = 4
    n_out = 2
    rng = np.random.default_rng(123)

    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            OffsetTransform(output_size=n_stars, vmap=False),
        )
    )

    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = rng.random((n_out, n_latents))

    # Test error when wrong number of parameter dictionaries
    with pytest.raises(ValueError, match="Expected 2 parameter dictionaries"):
        trans.apply(latents, {"A": A})  # Missing second dict

    # Test error for unsupported parameter naming
    with pytest.raises(ValueError, match="Unsupported parameter name format"):
        trans.apply(latents, invalid_param=A)


def test_numpyro_parameter_naming_integration():
    """Test integration with NumPyro parameter naming scheme."""
    n_stars = 32
    n_latents = 8
    n_out = 4

    # Create a simple model with TransformSequence
    model = plx.LuxModel(latent_size=n_latents)
    trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            OffsetTransform(
                output_size=n_out, vmap=False
            ),  # Fixed: should be n_out, not n_stars
        )
    )
    model.register_output("test", trans)

    # Test that get_expanded_priors returns flat priors with new naming
    priors = trans.get_expanded_priors(latent_size=n_latents, data_size=n_stars)

    expected_names = {"0:A", "1:b"}
    assert set(priors.keys()) == expected_names

    # Test shapes
    assert priors["0:A"].batch_shape == (n_out, n_latents)
    assert priors["1:b"].batch_shape == (n_out, 1)  # Fixed: should be (n_out, 1)


def test_function_transform_in_sequence():
    """Test FunctionTransform within TransformSequence with new parameter scheme."""
    n_stars = 16
    n_latents = 4
    n_flux = 8
    rng = np.random.default_rng(789)

    # Create function transform similar to the notebook example
    def custom_transform(x, p1, p2):
        return x + p1[:, None] * 0.5 + p2[:, None] * 0.25

    func_trans = FunctionTransform(
        output_size=n_flux,
        transform=custom_transform,
        param_priors={"p1": dist.Normal(0.0, 1.0), "p2": dist.Normal(0.0, 1.0)},
        param_shapes={"p1": (n_stars,), "p2": (n_stars,)},
        vmap=False,
    )

    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_flux),
            func_trans,
        )
    )

    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = rng.random((n_flux, n_latents))
    p1 = rng.random((n_stars,))
    p2 = rng.random((n_stars,))

    # Test with new parameter format using *args
    result = trans_seq.apply(latents, {"A": A}, {"p1": p1, "p2": p2})

    # Verify the computation manually
    intermediate = A @ latents.T  # Shape: (n_flux, n_stars)
    expected = intermediate.T + p1[:, None] * 0.5 + p2[:, None] * 0.25

    assert np.allclose(result, expected)

    # Test with flat parameter format
    result_flat = trans_seq.apply(latents, **{"0:A": A, "1:p1": p1, "1:p2": p2})

    assert np.allclose(result_flat, expected)
