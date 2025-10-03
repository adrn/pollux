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
    pars = {"mags": [{"A": A}, {"b": b}]}

    model = plx.LuxModel(latent_size=n_latents)
    model.register_output("mags", trans)
    out = model.predict_outputs(latents, pars)

    assert np.allclose(out["mags"], tmp)

    # Test new flat parameter format
    pars_flat = {"mags": {"0:A": A, "1:b": b}}
    out_flat = model.predict_outputs(latents, pars_flat)
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


def test_transform_sequence_pack_unpack_pars():
    """Test the pack_pars and unpack_pars methods for parameter conversion.

    Tests basic functionality of packing nested parameter lists into flat dictionaries
    and vice versa. Verifies round-trip conversion (pack → unpack → original structure)
    with a two-transform sequence (LinearTransform + OffsetTransform).

    Covers:
    - Packing: nested list → flat dict with "{index}:{param}" naming
    - Unpacking: flat dict → nested list structure
    - Round-trip conversion preserves all parameter data accurately
    """
    n_latents = 8
    n_out = 4
    rng = np.random.default_rng(456)

    # Create a TransformSequence with multiple transforms
    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            OffsetTransform(output_size=n_out, vmap=False),
        )
    )

    # Create test parameter values
    A = rng.random((n_out, n_latents))
    b = rng.random((n_out, 1))

    # Test packing: nested list -> flat dict
    nested_pars = [
        {"A": A},
        {"b": b},
    ]

    packed = trans_seq.pack_pars(nested_pars)
    expected_packed = {"0:A": A, "1:b": b}

    assert set(packed.keys()) == set(expected_packed.keys())
    assert np.allclose(packed["0:A"], expected_packed["0:A"])
    assert np.allclose(packed["1:b"], expected_packed["1:b"])

    # Test unpacking: flat dict -> nested list
    flat_pars = {"0:A": A, "1:b": b}
    unpacked = trans_seq.unpack_pars(flat_pars)

    assert len(unpacked) == 2
    assert set(unpacked[0].keys()) == {"A"}
    assert set(unpacked[1].keys()) == {"b"}
    assert np.allclose(unpacked[0]["A"], A)
    assert np.allclose(unpacked[1]["b"], b)

    # Test round-trip: pack -> unpack should return original
    round_trip = trans_seq.unpack_pars(trans_seq.pack_pars(nested_pars))
    assert len(round_trip) == len(nested_pars)
    for orig, restored in zip(nested_pars, round_trip):
        assert set(orig.keys()) == set(restored.keys())
        for key in orig:
            assert np.allclose(orig[key], restored[key])


def test_transform_sequence_unpack_with_missing_pars():
    """Test unpacking parameters when some parameters are missing.

    Tests behavior when only some transforms have parameters provided.
    Ensures empty dictionaries are created for transforms without parameters,
    demonstrating robust handling of partial parameter sets.

    Covers:
    - Graceful handling of missing parameters for some transforms
    - Empty dictionary creation for transforms without provided parameters
    - No errors when parameter coverage is incomplete
    """
    n_out = 4
    n_latents = 8
    rng = np.random.default_rng(123)

    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            OffsetTransform(output_size=n_out, vmap=False),
        )
    )

    # Only provide parameters for first transform
    flat_pars = {"0:A": rng.random((n_out, n_latents))}
    unpacked = trans_seq.unpack_pars(flat_pars, skip_missing=True)

    assert len(unpacked) == 2
    assert "A" in unpacked[0]
    assert len(unpacked[1]) == 0  # Second transform should have empty dict


def test_transform_sequence_unpack_with_extra_pars():
    """Test unpacking parameters with extra parameters that don't match any transform.

    Tests robustness when invalid parameter names are provided, including parameters
    with invalid transform indices and non-indexed parameter names. Ensures the
    unpacking process ignores invalid parameters gracefully without errors.

    Covers:
    - Invalid transform indices (beyond sequence length) are ignored
    - Non-indexed parameter names (missing ":") are ignored silently
    - Only valid parameters matching existing transforms are unpacked
    - Robust error handling for malformed parameter names
    """
    n_out = 4
    n_latents = 8
    rng = np.random.default_rng(789)

    trans_seq = TransformSequence(transforms=(LinearTransform(output_size=n_out),))

    # Include valid parameter and some invalid ones
    flat_pars = {
        "0:A": rng.random((n_out, n_latents)),
        "5:invalid": rng.random((2, 2)),  # Invalid transform index
        "not_indexed": rng.random((3, 3)),  # No index format
    }

    unpacked = trans_seq.unpack_pars(flat_pars)

    # Should only unpack valid parameters
    assert len(unpacked) == 1
    assert "A" in unpacked[0]
    # Invalid parameters should be ignored silently


def test_transform_sequence_pack_empty_dicts():
    """Test packing when some parameter dictionaries are empty.

    Tests the packing process when some transforms in the sequence have empty
    parameter dictionaries. Ensures that only non-empty parameters are included
    in the flat output dictionary, demonstrating efficient handling of sparse
    parameter sets.

    Covers:
    - Empty parameter dictionaries are handled gracefully
    - Only non-empty parameters appear in packed output
    - No spurious entries for transforms without parameters
    """
    n_out = 4
    rng = np.random.default_rng(456)

    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            OffsetTransform(output_size=n_out, vmap=False),
        )
    )

    # First transform has parameters, second is empty
    nested_pars = [
        {"A": rng.random((n_out, 4))},
        {},  # Empty dict for second transform
    ]

    packed = trans_seq.pack_pars(nested_pars)

    # Should only have parameters from first transform
    assert set(packed.keys()) == {"0:A"}


def test_transform_sequence_three_transforms_pack_unpack():
    """Test pack/unpack with three transforms to ensure indexing works correctly.

    Tests parameter conversion with a more complex three-transform sequence including
    a custom FunctionTransform. Verifies correct indexing with multiple transforms
    (0:A, 1:scale, 2:b) and tests the full round-trip conversion with complex
    parameter structures to ensure scalability.

    Covers:
    - Correct indexing scheme for sequences of any length (0, 1, 2, ...)
    - Integration with custom FunctionTransform parameters
    - Multiple parameter types (matrices, scalars, vectors)
    - Full round-trip conversion preserves complex parameter structures
    - Scalability beyond simple two-transform cases
    """
    n_latents = 4
    n_out = 3
    rng = np.random.default_rng(321)

    # Create function transform for middle position
    def simple_func(x, scale):
        return x * scale

    func_trans = FunctionTransform(
        output_size=n_out,
        transform=simple_func,
        param_priors={"scale": dist.Normal(1.0, 0.1)},
        param_shapes={"scale": (1,)},
        vmap=False,
    )

    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=n_out),
            func_trans,
            OffsetTransform(output_size=n_out, vmap=False),
        )
    )

    # Create parameters for all three transforms
    A = rng.random((n_out, n_latents))
    scale = rng.random((1,))
    b = rng.random((n_out, 1))

    nested_pars = [
        {"A": A},
        {"scale": scale},
        {"b": b},
    ]

    # Test pack -> unpack round trip
    packed = trans_seq.pack_pars(nested_pars)
    expected_keys = {"0:A", "1:scale", "2:b"}
    assert set(packed.keys()) == expected_keys

    unpacked = trans_seq.unpack_pars(packed)
    assert len(unpacked) == 3
    assert np.allclose(unpacked[0]["A"], A)
    assert np.allclose(unpacked[1]["scale"], scale)
    assert np.allclose(unpacked[2]["b"], b)
