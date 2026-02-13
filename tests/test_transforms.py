import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux.models.transforms import (
    AdditiveOffsetTransform,
    ConcatenateTransform,
    EquinoxNNTransform,
    FunctionTransform,
    LinearTransform,
    OffsetTransform,
    PolyFeatureTransform,
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
        output_size=n_out, priors={"A": dist.Normal(0.0, 1.0)}
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
        output_size=n_stars, vmap=False, priors={"b": dist.Normal(0.0, 1.0)}
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
            LinearTransform(output_size=8, priors={"A": dist.Laplace()}),
            OffsetTransform(
                output_size=n_stars,
                vmap=False,
                priors={"b": dist.Normal(11.0, 3.0)},
            ),
        )
    )

    latents = jnp.array(rng.random((n_stars, n_latents)))

    A = rng.random((n_out, n_latents))
    b = rng.random((n_stars, n_out))

    tmp = np.array([A @ latents[i] for i in range(n_stars)])
    tmp += b

    # Test nested parameter format with list of dicts
    pars = {"mags": {"data": [{"A": A}, {"b": b}]}}

    model = plx.LuxModel(latent_size=n_latents)
    model.register_output("mags", trans)
    out = model.predict_outputs(latents, pars)

    assert np.allclose(out["mags"], tmp)

    # Test nested parameter format with flat dict
    pars_flat = {"mags": {"data": {"0:A": A, "1:b": b}}}
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

    # Test that priors and shapes are properly stored as tuples
    assert len(trans.priors) == 2
    assert len(trans.shapes) == 2

    # First transform (LinearTransform) should have 'A' parameter
    assert "A" in trans.priors[0]
    assert "A" in trans.shapes[0]

    # Second transform (OffsetTransform) should have 'b' parameter
    assert "b" in trans.priors[1]
    assert "b" in trans.shapes[1]


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
        priors={"p1": dist.Normal(0.0, 1.0), "p2": dist.Normal(0.0, 1.0)},
        shapes={"p1": (n_stars,), "p2": (n_stars,)},
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
    unpacked = trans_seq.unpack_pars(flat_pars, ignore_missing=True)

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
        priors={"scale": dist.Normal(1.0, 0.1)},
        shapes={"scale": (1,)},
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


def test_parameter_name_with_colon_raises():
    """Test that parameter names containing colons are rejected."""

    # Define a custom transform function with a parameter name containing a colon
    def bad_transform(x, bad_param):
        return x + bad_param

    # This should raise because of the colon in priors key
    with pytest.raises(ValueError, match="contains ':'"):
        FunctionTransform(
            output_size=4,
            transform=bad_transform,
            priors={"bad:param": dist.Normal(0.0, 1.0)},
            shapes={"bad:param": (4,)},
        )


def test_poly_feature_transform_basic():
    """Test basic polynomial feature expansion."""
    # Test with 2 inputs, degree 2, with bias
    trans = PolyFeatureTransform(degree=2, include_bias=True)

    # Input: 2 samples, 2 features
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = trans.apply(x)

    # Expected features for degree=2 with bias:
    # [1, x1, x2, x1^2, x1*x2, x2^2]
    # For [1, 2]: [1, 1, 2, 1, 2, 4]
    # For [3, 4]: [1, 3, 4, 9, 12, 16]
    expected = jnp.array(
        [[1.0, 1.0, 2.0, 1.0, 2.0, 4.0], [1.0, 3.0, 4.0, 9.0, 12.0, 16.0]]
    )

    assert result.shape == (2, 6)
    assert np.allclose(result, expected)


def test_poly_feature_transform_no_bias():
    """Test polynomial feature expansion without bias."""
    trans = PolyFeatureTransform(degree=2, include_bias=False)

    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = trans.apply(x)

    # Expected features without bias: [x1, x2, x1^2, x1*x2, x2^2]
    expected = jnp.array([[1.0, 2.0, 1.0, 2.0, 4.0], [3.0, 4.0, 9.0, 12.0, 16.0]])

    assert result.shape == (2, 5)
    assert np.allclose(result, expected)


def test_poly_feature_transform_degree_1():
    """Test polynomial feature expansion with degree 1 (just linear features)."""
    trans = PolyFeatureTransform(degree=1, include_bias=True)

    x = jnp.array([[1.0, 2.0, 3.0]])
    result = trans.apply(x)

    # Expected: [1, x1, x2, x3]
    expected = jnp.array([[1.0, 1.0, 2.0, 3.0]])

    assert result.shape == (1, 4)
    assert np.allclose(result, expected)


def test_poly_feature_transform_get_output_size():
    """Test the output size computation."""
    trans = PolyFeatureTransform(degree=2, include_bias=True)

    # For 2 inputs with degree 2: C(2+2, 2) = C(4, 2) = 6
    assert trans.get_output_size(2) == 6

    # For 3 inputs with degree 2: C(3+2, 2) = C(5, 2) = 10
    assert trans.get_output_size(3) == 10

    # For 2 inputs with degree 3: C(2+3, 3) = C(5, 3) = 10
    trans_deg3 = PolyFeatureTransform(degree=3, include_bias=True)
    assert trans_deg3.get_output_size(2) == 10


def test_poly_feature_transform_no_learnable_params():
    """Test that PolyFeatureTransform has no learnable parameters."""
    trans = PolyFeatureTransform(degree=2)

    # priors should be empty
    assert len(trans.priors) == 0

    # get_expanded_priors should return empty
    priors = trans.get_expanded_priors(latent_size=8, data_size=100)
    assert len(priors) == 0


def test_poly_feature_transform_in_sequence():
    """Test PolyFeatureTransform in a TransformSequence (Cannon-style)."""
    n_stars = 32
    n_labels = 3
    n_flux = 16
    rng = np.random.default_rng(42)

    # Create Cannon-style transform: polynomial features -> linear
    cannon_trans = TransformSequence(
        transforms=(
            PolyFeatureTransform(degree=2, include_bias=True),
            LinearTransform(output_size=n_flux),
        )
    )

    # Generate mock labels
    labels = jnp.array(rng.random((n_stars, n_labels)))

    # Compute expected number of polynomial features
    n_poly_features = PolyFeatureTransform(degree=2).get_output_size(n_labels)
    assert n_poly_features == 10  # C(3+2, 2) = 10

    # Create coefficient matrix
    A = rng.random((n_flux, n_poly_features))

    # Apply transform with flat parameter format
    result = cannon_trans.apply(labels, **{"0:": {}, "1:A": A})

    # Actually the 0th transform has no params, so we should use simpler format
    # Let me try with the dict-based approach
    result = cannon_trans.apply(labels, {}, {"A": A})

    assert result.shape == (n_stars, n_flux)

    # Verify computation manually
    poly_features = PolyFeatureTransform(degree=2, include_bias=True).apply(labels)
    expected = (
        poly_features @ A.T
    )  # (n_stars, n_poly_features) @ (n_poly_features, n_flux).T
    assert np.allclose(result, expected)


def test_poly_feature_transform_with_lux_model():
    """Test PolyFeatureTransform integration with LuxModel."""

    n_stars = 16
    n_labels = 3
    n_flux = 8
    rng = np.random.default_rng(123)

    # Create model with Cannon-style transform
    model = plx.LuxModel(latent_size=n_labels)

    cannon_trans = TransformSequence(
        transforms=(
            PolyFeatureTransform(degree=2, include_bias=True),
            LinearTransform(output_size=n_flux),
        )
    )
    model.register_output("flux", cannon_trans)

    # Generate synthetic data
    labels = jnp.array(rng.random((n_stars, n_labels)))
    n_poly_features = PolyFeatureTransform(degree=2).get_output_size(n_labels)
    A = rng.random((n_flux, n_poly_features))

    # Compute true flux
    poly_features = PolyFeatureTransform(degree=2, include_bias=True).apply(labels)
    true_flux = poly_features @ A.T

    # Create data
    flux_data = plx.data.OutputData(data=true_flux, err=0.01 * np.ones_like(true_flux))
    plx.data.PolluxData(flux=flux_data)

    # Get priors
    priors = cannon_trans.get_expanded_priors(latent_size=n_labels, data_size=n_stars)

    # First transform (PolyFeatureTransform) has no priors
    # Second transform (LinearTransform) has "1:A" prior
    assert "1:A" in priors
    assert priors["1:A"].batch_shape == (n_flux, n_poly_features)

    # Test predict_outputs
    pars = {"flux": {"data": [{}, {"A": A}]}}
    result = model.predict_outputs(labels, pars)
    assert np.allclose(result["flux"], true_flux)


# ---- EquinoxNNTransform Tests ----


def test_equinox_nn_transform_basic():
    """Test basic EquinoxNNTransform functionality."""
    n_in = 4
    n_out = 8
    n_samples = 16

    # Create a simple MLP factory
    def mlp_factory(in_size, out_size, key):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=16,
            depth=1,
            key=key,
        )

    nn_trans = EquinoxNNTransform(
        output_size=n_out,
        nn_factory=mlp_factory,
        weight_prior=dist.Normal(0.0, 1.0),
        bias_prior=dist.Normal(0.0, 0.1),
    )

    # Get expanded priors
    priors = nn_trans.get_expanded_priors(latent_size=n_in)

    # Check that we have priors for all parameters
    assert len(priors) > 0

    # Check that weight and bias priors exist
    weight_paths = [p for p in priors if "weight" in p]
    bias_paths = [p for p in priors if "bias" in p]
    assert len(weight_paths) > 0
    assert len(bias_paths) > 0

    # Sample parameters from priors
    rng = np.random.default_rng(42)
    params = {
        path: rng.normal(size=prior.batch_shape) for path, prior in priors.items()
    }

    # Apply transform
    latents = jnp.array(rng.random((n_samples, n_in)))
    result = nn_trans.apply(latents, **params)

    assert result.shape == (n_samples, n_out)


def test_equinox_nn_transform_param_paths():
    """Test that parameter paths are correctly generated."""
    n_in = 4
    n_out = 8

    def mlp_factory(in_size, out_size, key):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=16,
            depth=2,  # 2 hidden layers
            key=key,
        )

    nn_trans = EquinoxNNTransform(
        output_size=n_out,
        nn_factory=mlp_factory,
    )

    priors = nn_trans.get_expanded_priors(latent_size=n_in)

    # For depth=2 MLP, we expect:
    # - layers.0.weight, layers.0.bias (input -> hidden1)
    # - layers.1.weight, layers.1.bias (hidden1 -> hidden2)
    # - layers.2.weight, layers.2.bias (hidden2 -> output)
    expected_paths = {
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
        "layers.2.weight",
        "layers.2.bias",
    }

    assert set(priors.keys()) == expected_paths


def test_equinox_nn_transform_prior_shapes():
    """Test that prior shapes are correct."""
    n_in = 4
    n_out = 8
    width = 16

    def mlp_factory(in_size, out_size, key):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=1,
            key=key,
        )

    nn_trans = EquinoxNNTransform(
        output_size=n_out,
        nn_factory=mlp_factory,
    )

    priors = nn_trans.get_expanded_priors(latent_size=n_in)

    # Check shapes
    # layers.0: input (4) -> hidden (16)
    assert priors["layers.0.weight"].batch_shape == (width, n_in)
    assert priors["layers.0.bias"].batch_shape == (width,)

    # layers.1: hidden (16) -> output (8)
    assert priors["layers.1.weight"].batch_shape == (n_out, width)
    assert priors["layers.1.bias"].batch_shape == (n_out,)


def test_equinox_nn_transform_custom_priors():
    """Test that custom priors are applied correctly."""
    n_in = 4
    n_out = 8

    def mlp_factory(in_size, out_size, key):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=16,
            depth=1,
            key=key,
        )

    # Use distinctive priors
    weight_prior = dist.Normal(0.0, 0.5)
    bias_prior = dist.Normal(1.0, 0.1)

    nn_trans = EquinoxNNTransform(
        output_size=n_out,
        nn_factory=mlp_factory,
        weight_prior=weight_prior,
        bias_prior=bias_prior,
    )

    priors = nn_trans.get_expanded_priors(latent_size=n_in)

    # Check that weight priors have correct parameters
    for path, prior in priors.items():
        if "weight" in path:
            assert prior.base_dist.loc == 0.0
            assert prior.base_dist.scale == 0.5
        elif "bias" in path:
            assert prior.base_dist.loc == 1.0
            assert prior.base_dist.scale == 0.1


def test_equinox_nn_transform_deterministic():
    """Test that apply produces deterministic output for same params."""
    n_in = 4
    n_out = 8
    n_samples = 16
    rng = np.random.default_rng(42)

    def mlp_factory(in_size, out_size, key):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=16,
            depth=1,
            key=key,
        )

    nn_trans = EquinoxNNTransform(
        output_size=n_out,
        nn_factory=mlp_factory,
    )

    priors = nn_trans.get_expanded_priors(latent_size=n_in)
    params = {
        path: jnp.array(rng.normal(size=prior.batch_shape))
        for path, prior in priors.items()
    }
    latents = jnp.array(rng.random((n_samples, n_in)))

    # Apply twice with same params
    result1 = nn_trans.apply(latents, **params)
    result2 = nn_trans.apply(latents, **params)

    assert np.allclose(result1, result2)


def test_equinox_nn_transform_with_lux_model():
    """Test EquinoxNNTransform integration with LuxModel."""
    n_stars = 16
    n_latents = 4
    n_flux = 8
    rng = np.random.default_rng(123)

    def mlp_factory(in_size, out_size, key):
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=16,
            depth=1,
            key=key,
        )

    # Create model with NN transform
    model = plx.LuxModel(latent_size=n_latents)
    nn_trans = EquinoxNNTransform(
        output_size=n_flux,
        nn_factory=mlp_factory,
        weight_prior=dist.Normal(0.0, 0.1),
        bias_prior=dist.Normal(0.0, 0.01),
    )
    model.register_output("flux", nn_trans)

    # Generate latents
    latents = jnp.array(rng.random((n_stars, n_latents)))

    # Get priors and sample parameters
    priors = nn_trans.get_expanded_priors(latent_size=n_latents, data_size=n_stars)
    params = {
        path: jnp.array(rng.normal(size=prior.batch_shape) * 0.1)
        for path, prior in priors.items()
    }

    # Test predict_outputs
    pars = {"flux": {"data": params}}
    result = model.predict_outputs(latents, pars)

    assert result["flux"].shape == (n_stars, n_flux)


# --- AdditiveOffsetTransform tests ---


def test_additive_offset_transform_basic():
    """Test basic AdditiveOffsetTransform functionality."""
    n_stars = 32
    n_latents = 8
    n_output = 4
    rng = np.random.default_rng(42)

    # Create transform with LinearTransform as base
    base_trans = LinearTransform(output_size=n_output)
    offset_trans = AdditiveOffsetTransform(
        base_transform=base_trans,
        offset_prior=dist.Normal(10.0, 2.0),
    )

    # Check output_size is inherited
    assert offset_trans.output_size == n_output

    # Generate data
    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = jnp.array(rng.random((n_output, n_latents)))
    offset = jnp.array(rng.normal(10.0, 2.0, size=(n_stars,)))

    # Apply transform with "base:" prefix for base params
    result = offset_trans.apply(latents, **{"base:A": A, "offset": offset})

    # Verify shape
    assert result.shape == (n_stars, n_output)

    # Verify computation: base_output + offset[:, None]
    base_output = base_trans.apply(latents, A=A)
    expected = base_output + offset[:, None]
    assert np.allclose(result, expected)


def test_additive_offset_transform_priors():
    """Test that AdditiveOffsetTransform generates correct priors."""
    n_latents = 8
    n_output = 4
    n_stars = 100

    base_trans = LinearTransform(output_size=n_output)
    offset_trans = AdditiveOffsetTransform(
        base_transform=base_trans,
        offset_prior=dist.Normal(11.0, 3.0),
    )

    # Get expanded priors
    priors = offset_trans.get_expanded_priors(latent_size=n_latents, data_size=n_stars)

    # Should have base:A and offset
    assert "base:A" in priors
    assert "offset" in priors

    # Check shapes
    assert priors["base:A"].batch_shape == (n_output, n_latents)
    assert priors["offset"].batch_shape == (n_stars,)

    # Check offset prior parameters (access base distribution)
    assert priors["offset"].base_dist.loc == 11.0
    assert priors["offset"].base_dist.scale == 3.0


def test_additive_offset_transform_requires_data_size():
    """Test that AdditiveOffsetTransform raises error without data_size."""
    base_trans = LinearTransform(output_size=4)
    offset_trans = AdditiveOffsetTransform(base_transform=base_trans)

    with pytest.raises(ValueError, match="requires data_size"):
        offset_trans.get_expanded_priors(latent_size=8, data_size=None)


def test_additive_offset_transform_pack_unpack():
    """Test parameter packing and unpacking."""
    n_output = 4
    n_latents = 8
    n_stars = 32
    rng = np.random.default_rng(42)

    base_trans = LinearTransform(output_size=n_output)
    offset_trans = AdditiveOffsetTransform(base_transform=base_trans)

    # Create flat parameters
    A = jnp.array(rng.random((n_output, n_latents)))
    offset = jnp.array(rng.random((n_stars,)))
    flat_pars = {"base:A": A, "offset": offset}

    # Unpack
    nested = offset_trans.unpack_pars(flat_pars)
    assert "base" in nested
    assert "offset" in nested
    assert "A" in nested["base"]
    assert np.allclose(nested["base"]["A"], A)
    assert np.allclose(nested["offset"], offset)

    # Pack back
    repacked = offset_trans.pack_pars(nested)
    assert "base:A" in repacked
    assert "offset" in repacked
    assert np.allclose(repacked["base:A"], A)
    assert np.allclose(repacked["offset"], offset)


def test_additive_offset_transform_different_data_sizes():
    """Test that offset adapts to different data sizes."""
    n_latents = 8
    n_output = 4

    base_trans = LinearTransform(output_size=n_output)
    offset_trans = AdditiveOffsetTransform(
        base_transform=base_trans,
        offset_prior=dist.Normal(11.0, 3.0),
    )

    # Training size
    priors_train = offset_trans.get_expanded_priors(
        latent_size=n_latents, data_size=1000
    )
    assert priors_train["offset"].batch_shape == (1000,)

    # Test size (different)
    priors_test = offset_trans.get_expanded_priors(latent_size=n_latents, data_size=500)
    assert priors_test["offset"].batch_shape == (500,)


def test_additive_offset_transform_with_lux_model():
    """Test AdditiveOffsetTransform integration with Lux model."""

    n_stars = 32
    n_latents = 8
    n_output = 3
    rng = np.random.default_rng(42)

    # Create model with AdditiveOffsetTransform
    model = plx.Lux(latent_size=n_latents)
    offset_trans = AdditiveOffsetTransform(
        base_transform=LinearTransform(output_size=n_output),
        offset_prior=dist.Normal(11.0, 3.0),
    )
    model.register_output("phot", offset_trans)

    # Generate data
    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = jnp.array(rng.random((n_output, n_latents)))
    offset = jnp.array(rng.normal(11.0, 3.0, size=(n_stars,)))

    # Test predict_outputs
    pars = {"phot": {"data": {"base:A": A, "offset": offset}}}
    result = model.predict_outputs(latents, pars)

    assert result["phot"].shape == (n_stars, n_output)

    # Verify computation
    base_output = jnp.einsum("ij,nj->ni", A, latents)
    expected = base_output + offset[:, None]
    assert np.allclose(result["phot"], expected, atol=1e-5)


def test_additive_offset_transform_broadcast():
    """Test that offset broadcasts correctly to all output dimensions."""
    n_stars = 10
    n_latents = 4
    n_output = 5
    rng = np.random.default_rng(42)

    base_trans = LinearTransform(output_size=n_output)
    offset_trans = AdditiveOffsetTransform(base_transform=base_trans)

    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = jnp.array(rng.random((n_output, n_latents)))
    # Single scalar offset per star
    offset = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    result = offset_trans.apply(latents, **{"base:A": A, "offset": offset})
    base_output = base_trans.apply(latents, A=A)

    # Each row should have the same offset added to all elements
    for i in range(n_stars):
        for j in range(n_output):
            expected_val = base_output[i, j] + offset[i]
            assert np.isclose(result[i, j], expected_val)


# --------------------------------------------------------------------------
# ConcatenateTransform tests
# --------------------------------------------------------------------------


class TestConcatenateTransformValidation:
    """Tests for ConcatenateTransform initialization validation."""

    def test_empty_transforms_raises(self):
        with pytest.raises(Exception, match="At least one transform required"):
            ConcatenateTransform(transforms=(), input_sizes=())

    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception, match="must match"):
            ConcatenateTransform(
                transforms=(LinearTransform(output_size=4),),
                input_sizes=(3, 4),
            )

    def test_output_size_is_sum(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=10),
                LinearTransform(output_size=4),
            ),
            input_sizes=(3, 5),
        )
        assert concat.output_size == 14


class TestConcatenateTransformApply:
    """Tests for ConcatenateTransform.apply."""

    def test_basic_apply_kwargs(self):
        rng = np.random.default_rng(42)
        n_stars = 16
        in1, in2 = 3, 4
        out1, out2 = 5, 6

        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=out1),
                LinearTransform(output_size=out2),
            ),
            input_sizes=(in1, in2),
        )

        latents = jnp.array(rng.random((n_stars, in1 + in2)))
        A0 = jnp.array(rng.random((out1, in1)))
        A1 = jnp.array(rng.random((out2, in2)))

        result = concat.apply(latents, **{"0:A": A0, "1:A": A1})

        # Manually compute expected output
        z0 = latents[:, :in1]
        z1 = latents[:, in1:]
        expected = jnp.concatenate(
            [jnp.einsum("ij,nj->ni", A0, z0), jnp.einsum("ij,nj->ni", A1, z1)],
            axis=-1,
        )
        assert np.allclose(result, expected, atol=1e-5)

    def test_basic_apply_positional_args(self):
        rng = np.random.default_rng(0)
        n_stars = 8
        in1, in2 = 2, 3
        out1, out2 = 4, 5

        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=out1),
                LinearTransform(output_size=out2),
            ),
            input_sizes=(in1, in2),
        )

        latents = jnp.array(rng.random((n_stars, in1 + in2)))
        A0 = jnp.array(rng.random((out1, in1)))
        A1 = jnp.array(rng.random((out2, in2)))

        result = concat.apply(latents, {"A": A0}, {"A": A1})

        z0 = latents[:, :in1]
        z1 = latents[:, in1:]
        expected = jnp.concatenate(
            [jnp.einsum("ij,nj->ni", A0, z0), jnp.einsum("ij,nj->ni", A1, z1)],
            axis=-1,
        )
        assert np.allclose(result, expected, atol=1e-5)

    def test_apply_wrong_n_positional_args(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=4),
                LinearTransform(output_size=4),
            ),
            input_sizes=(3, 3),
        )
        latents = jnp.zeros((2, 6))
        with pytest.raises(ValueError, match="Expected 2 parameter dictionaries"):
            concat.apply(latents, {"A": jnp.zeros((4, 3))})


class TestConcatenateTransformPriors:
    """Tests for ConcatenateTransform.get_expanded_priors and naming."""

    def test_get_expanded_priors_flat_names(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(3, 4),
        )

        priors = concat.get_expanded_priors(latent_size=7)
        assert set(priors.keys()) == {"0:A", "1:A"}
        assert priors["0:A"].batch_shape == (5, 3)
        assert priors["1:A"].batch_shape == (6, 4)

    def test_latent_size_mismatch_raises(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(3, 4),
        )
        with pytest.raises(Exception, match="does not match"):
            concat.get_expanded_priors(latent_size=10)

    def test_names_flat(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(3, 4),
        )
        assert concat.names_flat == ("0:A", "1:A")

    def test_param_names_property(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(3, 4),
        )
        assert concat._param_names == ("0:A", "1:A")


class TestConcatenateTransformPackUnpack:
    """Tests for pack/unpack round-trip."""

    def test_pack_unpack_roundtrip(self):
        rng = np.random.default_rng(42)

        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(3, 4),
        )

        A0 = jnp.array(rng.random((5, 3)))
        A1 = jnp.array(rng.random((6, 4)))

        flat = {"0:A": A0, "1:A": A1}
        nested = concat.unpack_pars(flat)
        assert len(nested) == 2
        assert np.allclose(nested[0]["A"], A0)
        assert np.allclose(nested[1]["A"], A1)

        repacked = concat.pack_pars(list(nested))
        assert set(repacked.keys()) == set(flat.keys())
        for k in flat:
            assert np.allclose(repacked[k], flat[k])


class TestConcatenateTransformGetOutputSize:
    """Tests for get_output_size."""

    def test_returns_total_output_size(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=10),
                LinearTransform(output_size=4),
            ),
            input_sizes=(3, 5),
        )
        assert concat.get_output_size(8) == 14

    def test_wrong_input_size_raises(self):
        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=10),
                LinearTransform(output_size=4),
            ),
            input_sizes=(3, 5),
        )
        with pytest.raises(Exception, match="does not match"):
            concat.get_output_size(99)


class TestConcatenateTransformWithPolyFeature:
    """Tests for ConcatenateTransform with PolyFeatureTransform child."""

    def test_with_poly_feature_in_sequence(self):
        """ConcatenateTransform wrapping a TransformSequence(Poly -> Linear)."""
        rng = np.random.default_rng(42)
        n_stars = 16
        in1, in2 = 3, 4
        out1, out2 = 10, 6

        concat = ConcatenateTransform(
            transforms=(
                TransformSequence(
                    (
                        PolyFeatureTransform(degree=2),
                        LinearTransform(output_size=out1),
                    )
                ),
                LinearTransform(output_size=out2),
            ),
            input_sizes=(in1, in2),
        )

        # PolyFeatureTransform(degree=2) on 3 inputs gives 10 features
        # (1 + 3 + 6 = 10 with bias), so inner LinearTransform input is 10
        priors = concat.get_expanded_priors(latent_size=in1 + in2)
        assert "0:1:A" in priors  # TransformSequence nests: "0:1:A"
        assert "1:A" in priors
        assert priors["0:1:A"].batch_shape == (out1, 10)
        assert priors["1:A"].batch_shape == (out2, in2)

        # Apply with concrete parameters
        A_inner = jnp.array(rng.random((out1, 10)))
        A_outer = jnp.array(rng.random((out2, in2)))
        latents = jnp.array(rng.random((n_stars, in1 + in2)))

        result = concat.apply(latents, **{"0:1:A": A_inner, "1:A": A_outer})
        assert result.shape == (n_stars, out1 + out2)


class TestConcatenateTransformInTransformSequence:
    """Tests for ConcatenateTransform nested inside TransformSequence."""

    def test_concat_inside_sequence(self):
        """ConcatenateTransform as the first step of a TransformSequence."""
        rng = np.random.default_rng(0)
        n_stars = 8
        in1, in2 = 3, 4
        concat_out = 5 + 6  # 11

        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(in1, in2),
        )

        seq = TransformSequence(
            (
                concat,
                LinearTransform(output_size=8),
            )
        )

        priors = seq.get_expanded_priors(latent_size=in1 + in2)
        # ConcatenateTransform params are prefixed "0:{concat_flat_name}"
        assert "0:0:A" in priors
        assert "0:1:A" in priors
        # Second transform in sequence: "1:A"
        assert "1:A" in priors
        # Second linear takes concat output (11) as input
        assert priors["1:A"].batch_shape == (8, concat_out)

        A0 = jnp.array(rng.random((5, in1)))
        A1 = jnp.array(rng.random((6, in2)))
        A2 = jnp.array(rng.random((8, concat_out)))

        latents = jnp.array(rng.random((n_stars, in1 + in2)))
        result = seq.apply(latents, **{"0:0:A": A0, "0:1:A": A1, "1:A": A2})
        assert result.shape == (n_stars, 8)


class TestConcatenateTransformWithLuxModel:
    """Tests for ConcatenateTransform with LuxModel integration."""

    def test_register_and_predict(self):
        rng = np.random.default_rng(42)
        n_stars = 16
        in1, in2 = 3, 4
        n_latents = in1 + in2

        concat = ConcatenateTransform(
            transforms=(
                LinearTransform(output_size=5),
                LinearTransform(output_size=6),
            ),
            input_sizes=(in1, in2),
        )

        model = plx.LuxModel(latent_size=n_latents)
        model.register_output("flux", concat)

        A0 = jnp.array(rng.random((5, in1)))
        A1 = jnp.array(rng.random((6, in2)))
        latents = jnp.array(rng.random((n_stars, n_latents)))

        pars = {"flux": {"data": {"0:A": A0, "1:A": A1}}}
        result = model.predict_outputs(latents, pars)
        assert result["flux"].shape == (n_stars, 11)


class TestParamNamesOnPolyFeature:
    """Verify _param_names is available on PolyFeatureTransform."""

    def test_param_names_empty(self):
        poly = PolyFeatureTransform(degree=2)
        assert poly._param_names == ()

    def test_poly_in_transform_sequence_names_nested(self):
        seq = TransformSequence(
            (
                PolyFeatureTransform(degree=2),
                LinearTransform(output_size=8),
            )
        )
        assert seq.names_nested == ((), ("A",))
        assert seq.names_flat == ("1:A",)
