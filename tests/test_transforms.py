from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux.models.transforms import (
    ConcatenateTransform,
    EquinoxNNTransform,
    FunctionTransform,
    LinearTransform,
    OffsetTransform,
    PolyFeatureTransform,
    ScatterTransform,
    TransformSequence,
)


def mlp_factory(in_size, out_size, key, width=16, depth=1):
    """Network factory for the EquinoxNNTransform tests."""
    return eqx.nn.MLP(
        in_size=in_size, out_size=out_size, width_size=width, depth=depth, key=key
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


def test_scatter_transform():
    n_stars = 64
    n_out = 8
    rng = np.random.default_rng(42)

    trans = ScatterTransform(output_size=n_out)
    err = jnp.array(rng.random((n_stars, n_out)))
    s = jnp.array(rng.random(n_out))

    # Adding the scatter in quadrature can only inflate the reported errors
    result = trans.apply(err, s=s)
    assert np.allclose(result, np.sqrt(np.array(err) ** 2 + np.array(s) ** 2))
    assert np.all(result >= err)

    # A zero scatter leaves the errors untouched
    assert np.allclose(trans.apply(err, s=jnp.zeros(n_out)), err)

    # The prior scale is a plain field, so it can be widened at construction
    wide = ScatterTransform(output_size=n_out, priors={"s": dist.HalfNormal(5.0)})
    assert np.allclose(wide.apply(err, s=s), result)
    assert wide.get_expanded_priors(latent_size=4, data_size=n_stars)[
        "s"
    ].batch_shape == (n_out,)


def test_scatter_transform_as_err_transform():
    """The point of the transform: an output whose scatter is fitted alongside it."""
    n_stars, n_latents, n_flux = 32, 2, 6
    rng = np.random.default_rng(0)

    model = plx.LVM(latent_size=n_latents)
    model.register_output(
        "flux",
        LinearTransform(output_size=n_flux),
        err_transform=ScatterTransform(output_size=n_flux),
    )

    latents = jnp.array(rng.normal(size=(n_stars, n_latents)))
    A = jnp.array(rng.normal(size=(n_flux, n_latents)))
    data = plx.data.PolluxData(
        flux=plx.data.OutputData(latents @ A.T, err=jnp.full((n_stars, n_flux), 0.05))
    )

    result = model.optimize_iterative(
        data, max_cycles=1, rng_key=jax.random.PRNGKey(0), progress=False
    )
    assert result.params["flux"]["err"]["s"].shape == (n_flux,)
    assert np.all(np.asarray(result.params["flux"]["err"]["s"]) >= 0)


@pytest.mark.parametrize(
    ("cls", "extra_pars"),
    [
        (plx.models.transforms.AffineTransform, {}),
        (plx.models.transforms.QuadraticTransform, {"Q": jnp.zeros((3, 4, 4))}),
    ],
)
def test_bias_shape_does_not_broadcast_to_a_matrix(cls, extra_pars):
    """The bias is one value per output dimension, not an (output_size, 1) column.

    With a trailing 1 the per-sample ``A @ z + b`` broadcasts ``(O,) + (O, 1)`` to
    ``(O, O)``, which made these transforms unusable in any model.
    """
    n_stars, n_latents, n_out = 6, 4, 3
    trans = cls(output_size=n_out)
    assert trans.shapes["b"] == ("output_size",)

    priors = trans.get_expanded_priors(latent_size=n_latents)
    assert priors["b"].batch_shape == (n_out,)

    out = trans.apply(
        jnp.ones((n_stars, n_latents)),
        A=jnp.ones((n_out, n_latents)),
        b=jnp.ones((n_out,)),
        **extra_pars,
    )
    assert out.shape == (n_stars, n_out)


@pytest.mark.parametrize(
    "cls",
    [plx.models.transforms.AffineTransform, plx.models.transforms.QuadraticTransform],
)
def test_bias_transforms_run_in_a_model(cls):
    """End-to-end guard: a model containing one of these used to fail to optimize."""
    n_stars, n_out = 6, 3
    model = plx.LVM(latent_size=4)
    model.register_output("flux", cls(output_size=n_out))
    data = plx.data.PolluxData(
        flux=plx.data.OutputData(
            data=jnp.ones((n_stars, n_out)), err=jnp.full((n_stars, n_out), 0.1)
        )
    )
    pars, _ = model.optimize(
        data,
        num_steps=2,
        rng_key=jax.random.PRNGKey(0),
        svi_run_kwargs={"progress_bar": False},
    )
    assert pars["flux"]["data"]["b"].shape == (n_out,)


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

    model = plx.LVM(latent_size=n_latents)
    model.register_output("mags", trans)
    out = model.predict_outputs(pars, latents)

    assert np.allclose(out["mags"], tmp)

    # Test nested parameter format with flat dict
    pars_flat = {"mags": {"data": {"0:A": A, "1:b": b}}}
    out_flat = model.predict_outputs(pars_flat, latents)
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
    model = plx.LVM(latent_size=n_latents)
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
    """Test pack/unpack round-tripping for a two-transform sequence."""
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
    """Unpacking with ignore_missing gives empty dicts for absent parameters."""
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
    """Unpacking ignores out-of-range indices and non-indexed parameter names."""
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
    """Packing a transform with an empty parameter dict adds no entries."""
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
    """Test pack/unpack indexing (0:A, 1:scale, 2:b) with three transforms."""
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
    """Test PolyFeatureTransform integration with LVM."""

    n_stars = 16
    n_labels = 3
    n_flux = 8
    rng = np.random.default_rng(123)

    # Create model with Cannon-style transform
    model = plx.LVM(latent_size=n_labels)

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
    result = model.predict_outputs(pars, labels)
    assert np.allclose(result["flux"], true_flux)


# ---- EquinoxNNTransform Tests ----


def test_equinox_nn_transform_basic():
    """Test basic EquinoxNNTransform functionality."""
    n_in = 4
    n_out = 8
    n_samples = 16

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

    nn_trans = EquinoxNNTransform(
        output_size=n_out,
        nn_factory=partial(mlp_factory, depth=2),  # 2 hidden layers
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
    """Test EquinoxNNTransform integration with LVM."""
    n_stars = 16
    n_latents = 4
    n_flux = 8
    rng = np.random.default_rng(123)

    # Create model with NN transform
    model = plx.LVM(latent_size=n_latents)
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
    result = model.predict_outputs(pars, latents)

    assert result["flux"].shape == (n_stars, n_flux)


# --- per-object offsets (the AdditiveOffsetTransform recipe) ---


def per_object_offset(output_size, offset_prior=None):
    """One scalar offset per object, added to every output dimension."""
    return FunctionTransform(
        output_size=output_size,
        transform=lambda x, offset: x + offset[:, None],
        priors={"offset": offset_prior or dist.Normal(11.0, 3.0)},
        shapes={"offset": ("data_size",)},
        vmap=False,
    )


def test_per_object_offset_applies_and_broadcasts():
    """A per-object offset is added to every output dimension of every object."""
    n_stars, n_latents, n_output = 10, 4, 5
    rng = np.random.default_rng(42)

    base = LinearTransform(output_size=n_output)
    trans = TransformSequence((base, per_object_offset(n_output)))

    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = jnp.array(rng.random((n_output, n_latents)))
    offset = jnp.arange(1.0, n_stars + 1.0)

    result = trans.apply(latents, **{"0:A": A, "1:offset": offset})
    base_output = base.apply(latents, A=A)

    assert result.shape == (n_stars, n_output)
    for i in range(n_stars):
        for j in range(n_output):
            assert np.isclose(result[i, j], base_output[i, j] + offset[i])


def test_per_object_offset_priors_track_data_size():
    """The offset prior is expanded to one value per object in the dataset."""
    n_latents, n_output = 8, 4
    trans = TransformSequence(
        (LinearTransform(output_size=n_output), per_object_offset(n_output))
    )

    for data_size in (1000, 500):
        priors = trans.get_expanded_priors(latent_size=n_latents, data_size=data_size)
        assert set(priors) == {"0:A", "1:offset"}
        assert priors["0:A"].batch_shape == (n_output, n_latents)
        assert priors["1:offset"].batch_shape == (data_size,)
        assert priors["1:offset"].base_dist.loc == 11.0
        assert priors["1:offset"].base_dist.scale == 3.0


def test_per_object_offset_requires_data_size():
    """Without a data size there is no way to shape the offset: say so clearly."""
    trans = TransformSequence((LinearTransform(output_size=4), per_object_offset(4)))
    with pytest.raises(ValueError, match="data_size"):
        trans.get_expanded_priors(latent_size=8, data_size=None)


def test_per_object_offset_with_lux_model():
    """The composed transform round-trips through a LVM model."""
    n_stars, n_latents, n_output = 32, 8, 3
    rng = np.random.default_rng(42)

    trans = TransformSequence(
        (LinearTransform(output_size=n_output), per_object_offset(n_output))
    )
    model = plx.LVM(latent_size=n_latents)
    model.register_output("phot", trans)

    latents = jnp.array(rng.random((n_stars, n_latents)))
    A = jnp.array(rng.random((n_output, n_latents)))
    offset = jnp.array(rng.normal(11.0, 3.0, size=(n_stars,)))

    pars = {"phot": {"data": {"0:A": A, "1:offset": offset}}}
    result = model.predict_outputs(pars, latents)

    expected = jnp.einsum("ij,nj->ni", A, latents) + offset[:, None]
    assert result["phot"].shape == (n_stars, n_output)
    assert np.allclose(result["phot"], expected, atol=1e-5)

    # ...and the packed parameter names survive a pack/unpack round trip
    packed = model.pack_numpyro_pars({"phot": {"data": [{"A": A}, {"offset": offset}]}})
    assert set(packed) == {"phot:0:A", "phot:1:offset"}
    unpacked = model.unpack_numpyro_pars(packed)
    assert np.allclose(unpacked["phot"]["data"][1]["offset"], offset)


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


class TestConcatenateTransformWithLux:
    """Tests for ConcatenateTransform with LVM integration."""

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

        model = plx.LVM(latent_size=n_latents)
        model.register_output("flux", concat)

        A0 = jnp.array(rng.random((5, in1)))
        A1 = jnp.array(rng.random((6, in2)))
        latents = jnp.array(rng.random((n_stars, n_latents)))

        pars = {"flux": {"data": {"0:A": A0, "1:A": A1}}}
        result = model.predict_outputs(pars, latents)
        assert result["flux"].shape == (n_stars, 11)


class TestPublicSignatures:
    """The keyword names of the public pack/unpack API."""

    def test_pack_pars_accepts_its_documented_keyword(self):
        """pack_pars is called with nested_pars= by name, not only positionally."""
        rng = np.random.default_rng(0)
        A = jnp.array(rng.random((3, 2)))

        single = LinearTransform(output_size=3)
        assert single.pack_pars(nested_pars={"A": A}) == {"A": A}

        seq = TransformSequence((single, OffsetTransform(output_size=3)))
        packed = seq.pack_pars(nested_pars=[{"A": A}, {}])
        assert set(packed) == {"0:A"}

        nn = EquinoxNNTransform(output_size=3, nn_factory=mlp_factory)
        nn.get_expanded_priors(latent_size=2)
        weight = jnp.zeros((16, 2))
        packed = nn.pack_pars(
            nested_pars={"layers.0.weight": weight}, ignore_missing=True
        )
        assert packed["layers.0.weight"] is weight


class TestEquinoxNNParamPaths:
    """Parameter paths for networks that are not plain attribute/sequence trees."""

    def test_dict_valued_module_fields(self):
        """A module holding submodules in a dict names its parameters, not crashes."""

        class DictNet(eqx.Module):
            layers: dict

            def __call__(self, x):
                return self.layers["out"](self.layers["in_"](x))

        def factory(in_size, out_size, key):
            k1, k2 = jax.random.split(key)
            return DictNet(
                layers={
                    "in_": eqx.nn.Linear(in_size, 8, key=k1),
                    "out": eqx.nn.Linear(8, out_size, key=k2),
                }
            )

        trans = EquinoxNNTransform(output_size=4, nn_factory=factory)
        priors = trans.get_expanded_priors(latent_size=3)

        assert set(priors) == {
            "layers.in_.weight",
            "layers.in_.bias",
            "layers.out.weight",
            "layers.out.bias",
        }
        assert priors["layers.in_.weight"].batch_shape == (8, 3)

        # ...and the network still runs with those parameters
        rng = np.random.default_rng(0)
        params = {
            path: jnp.array(rng.normal(size=prior.batch_shape))
            for path, prior in priors.items()
        }
        out = trans.apply(jnp.array(rng.random((5, 3))), **params)
        assert out.shape == (5, 4)


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
