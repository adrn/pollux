from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.optim
import pytest
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value

import pollux as plx
from pollux.models.transforms import (
    FunctionTransform,
    LinearTransform,
    NoOpTransform,
    OffsetTransform,
    TransformSequence,
)


@pytest.fixture
def rng():
    """Random number generator for consistent test results."""
    return np.random.default_rng(42)


@pytest.fixture
def model_config():
    """Basic model configuration used across tests."""
    return {
        "n_latents": 4,
        "n_flux": 8,
        "n_labels": 2,
        "n_stars": 16,
    }


@pytest.fixture
def single_transform_model(model_config):
    """LVM with a single LinearTransform for testing basic functionality."""
    model = plx.LVM(latent_size=model_config["n_latents"])
    model.register_output("flux", LinearTransform(output_size=model_config["n_flux"]))
    return model


@pytest.fixture
def single_transform_with_err_model(model_config):
    """LVM with LinearTransform and OffsetTransform error transform."""
    model = plx.LVM(latent_size=model_config["n_latents"])
    model.register_output(
        "flux",
        LinearTransform(output_size=model_config["n_flux"]),
        err_transform=OffsetTransform(output_size=model_config["n_flux"]),
    )
    return model


@pytest.fixture
def transform_sequence_model(model_config):
    """LVM with a TransformSequence (LinearTransform + OffsetTransform)."""
    model = plx.LVM(latent_size=model_config["n_latents"])
    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=model_config["n_flux"]),
            OffsetTransform(output_size=model_config["n_flux"]),
        )
    )
    model.register_output("flux", trans_seq)
    return model


@pytest.fixture
def transform_sequence_with_err_model(model_config):
    """LVM with TransformSequence and a FunctionTransform error transform."""
    model = plx.LVM(latent_size=model_config["n_latents"])

    # Data transform: sequence of LinearTransform + OffsetTransform
    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=model_config["n_flux"]),
            OffsetTransform(output_size=model_config["n_flux"]),
        )
    )

    # Error transform: simple scaling function
    def scale_func(x, scale):
        return x * scale

    err_trans = FunctionTransform(
        output_size=model_config["n_flux"],
        transform=scale_func,
        priors={"scale": dist.Normal(1.0, 0.1)},
        shapes={"scale": (1,)},
        vmap=False,
    )

    model.register_output("flux", trans_seq, err_transform=err_trans)
    return model


@pytest.fixture
def multi_output_model(model_config):
    """LVM with multiple outputs: TransformSequence flux + single transform labels."""
    model = plx.LVM(latent_size=model_config["n_latents"])

    # Flux output: TransformSequence
    flux_trans = TransformSequence(
        transforms=(
            LinearTransform(output_size=model_config["n_flux"]),
            OffsetTransform(output_size=model_config["n_flux"]),
        )
    )
    model.register_output("flux", flux_trans)

    # Label output: Single transform
    model.register_output(
        "label", LinearTransform(output_size=model_config["n_labels"])
    )
    return model


class TestLVMParameterPackUnpack:
    """Test suite for LVM parameter packing and unpacking functionality.

    These tests verify the new two-dictionary parameter structure that separates
    data transform parameters from error transform parameters. This design:
    1. Prevents parameter name conflicts between data and error transforms
    2. Maintains clean separation of concerns
    3. Supports nested parameter structures for TransformSequence
    4. Provides round-trip conversion integrity
    """

    def test_single_transform_basic(self, single_transform_model, model_config, rng):
        """Test basic pack/unpack functionality with a single LinearTransform.

        This test verifies the fundamental parameter conversion for the simplest case:
        a single transform with no error parameters. It ensures that:
        - Parameters are correctly packed into numpyro format ("flux:A")
        - Parameters are correctly unpacked back to nested structure ({"flux": {"A": ...}})
        - Round-trip conversion preserves data integrity
        - Error parameters dictionary is properly initialized but empty
        """
        # Create test parameters using output-centric structure (err key optional)
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))
        pars = {"flux": {"data": {"A": A}}}

        # Test packing
        packed = single_transform_model.pack_numpyro_pars(pars)
        expected_packed = {"flux:A": A}

        assert set(packed.keys()) == set(expected_packed.keys())
        assert np.allclose(packed["flux:A"], expected_packed["flux:A"])

        # Test unpacking
        unpacked = single_transform_model.unpack_numpyro_pars(packed)

        assert set(unpacked.keys()) == {"flux"}
        assert set(unpacked["flux"].keys()) == {"data", "err"}
        assert set(unpacked["flux"]["data"].keys()) == {"A"}
        assert np.allclose(unpacked["flux"]["data"]["A"], A)
        assert unpacked["flux"]["err"] == {}

    def test_err_key_optional(self, single_transform_model, model_config, rng):
        """Test that the 'err' key is optional when packing parameters.

        This test verifies that when a model has no error parameters, users don't
        need to specify an empty "err" key in the parameter structure. This makes
        the API more convenient for the common case where error transforms are not used.
        """
        # Create test parameters WITHOUT the "err" key
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))
        pars_without_err = {"flux": {"data": {"A": A}}}

        # Test packing - should work without "err" key
        packed = single_transform_model.pack_numpyro_pars(pars_without_err)
        expected_packed = {"flux:A": A}

        assert set(packed.keys()) == set(expected_packed.keys())
        assert np.allclose(packed["flux:A"], expected_packed["flux:A"])

        # Unpacking should still include "err" key (even if empty)
        unpacked = single_transform_model.unpack_numpyro_pars(packed)
        assert "err" in unpacked["flux"]
        assert unpacked["flux"]["err"] == {}

    def test_single_transform_with_error_pars(
        self, single_transform_with_err_model, model_config, rng
    ):
        """Test pack/unpack with single transform that has error transform parameters.

        This test validates parameter handling when both data and error transforms are present
        but both are single transforms (not sequences). It verifies:
        - Data parameters use standard naming ("flux:A")
        - Error parameters use prefixed naming ("flux:err:b")
        - Both parameter types are correctly separated during unpacking
        - No parameter name conflicts occur between data and error transforms
        """
        # Create test parameters using output-centric structure
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))
        b = rng.random((model_config["n_flux"], 1))
        pars = {"flux": {"data": {"A": A}, "err": {"b": b}}}

        # Test packing
        packed = single_transform_with_err_model.pack_numpyro_pars(pars)
        expected_keys = {"flux:A", "flux:err:b"}

        assert set(packed.keys()) == expected_keys
        assert np.allclose(packed["flux:A"], A)
        assert np.allclose(packed["flux:err:b"], b)

        # Test unpacking
        unpacked = single_transform_with_err_model.unpack_numpyro_pars(packed)

        assert set(unpacked.keys()) == {"flux"}
        assert set(unpacked["flux"].keys()) == {"data", "err"}
        assert set(unpacked["flux"]["data"].keys()) == {"A"}
        assert np.allclose(unpacked["flux"]["data"]["A"], A)

        assert set(unpacked["flux"]["err"].keys()) == {"b"}
        assert np.allclose(unpacked["flux"]["err"]["b"], b)

    def test_transform_sequence_without_error(
        self, transform_sequence_model, model_config, rng
    ):
        """Test pack/unpack with TransformSequence but no error transforms.

        This test focuses on the core TransformSequence parameter handling:
        - Data parameters use indexed naming ("flux:0:A", "flux:1:b")
        - Parameters are unpacked into a list structure for each transform
        - The list preserves the order and separation of transform parameters
        - Error parameters remain empty but properly structured
        """
        # Create test parameters using output-centric structure (err key optional)
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))
        b = rng.random((model_config["n_flux"], 1))
        pars = {"flux": {"data": [{"A": A}, {"b": b}]}}

        # Test packing
        packed = transform_sequence_model.pack_numpyro_pars(pars)
        expected_keys = {"flux:0:A", "flux:1:b"}

        assert set(packed.keys()) == expected_keys
        assert np.allclose(packed["flux:0:A"], A)
        assert np.allclose(packed["flux:1:b"], b)

        # Test unpacking
        unpacked = transform_sequence_model.unpack_numpyro_pars(packed)

        assert set(unpacked.keys()) == {"flux"}
        assert set(unpacked["flux"].keys()) == {"data", "err"}
        assert isinstance(unpacked["flux"]["data"], list | tuple)
        assert len(unpacked["flux"]["data"]) == 2

        assert set(unpacked["flux"]["data"][0].keys()) == {"A"}
        assert set(unpacked["flux"]["data"][1].keys()) == {"b"}
        assert np.allclose(unpacked["flux"]["data"][0]["A"], A)
        assert np.allclose(unpacked["flux"]["data"][1]["b"], b)

        assert unpacked["flux"]["err"] == {}

    def test_transform_sequence_with_error_pars(
        self, transform_sequence_with_err_model, model_config, rng
    ):
        """Test pack/unpack with both TransformSequence and error transform parameters.

        This test validates the most complex parameter scenario:
        - TransformSequence data parameters use indexed naming ("flux:0:A", "flux:1:b")
        - Error transform parameters use error-prefixed naming ("flux:err:scale")
        - Data parameters are unpacked to list structure
        - Error parameters are unpacked to flat dictionary structure
        - Both parameter types coexist without conflicts
        """
        # Create test parameters using output-centric structure
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))
        b = rng.random((model_config["n_flux"], 1))
        scale = rng.random((1,))

        pars = {"flux": {"data": [{"A": A}, {"b": b}], "err": {"scale": scale}}}

        # Test packing
        packed = transform_sequence_with_err_model.pack_numpyro_pars(pars)
        expected_keys = {"flux:0:A", "flux:1:b", "flux:err:scale"}

        assert set(packed.keys()) == expected_keys
        assert np.allclose(packed["flux:0:A"], A)
        assert np.allclose(packed["flux:1:b"], b)
        assert np.allclose(packed["flux:err:scale"], scale)

        # Test unpacking
        unpacked = transform_sequence_with_err_model.unpack_numpyro_pars(packed)

        # Check structure and data parameters (list structure)
        assert set(unpacked.keys()) == {"flux"}
        assert set(unpacked["flux"].keys()) == {"data", "err"}
        assert isinstance(unpacked["flux"]["data"], list | tuple)
        assert len(unpacked["flux"]["data"]) == 2
        assert np.allclose(unpacked["flux"]["data"][0]["A"], A)
        assert np.allclose(unpacked["flux"]["data"][1]["b"], b)

        # Check error parameters (flat structure)
        assert set(unpacked["flux"]["err"].keys()) == {"scale"}
        assert np.allclose(unpacked["flux"]["err"]["scale"], scale)

    def test_multiple_outputs_mixed_types(self, multi_output_model, model_config, rng):
        """Test pack/unpack with multiple outputs having different transform types.

        This test ensures the parameter system correctly handles models with mixed
        output types simultaneously:
        - One output uses TransformSequence (flux) with list-based parameters
        - Another output uses single transform (label) with dict-based parameters
        - Parameters for different outputs don't interfere with each other
        - The naming scheme correctly distinguishes between outputs
        """
        # Create test parameters (err key optional)
        A_flux = rng.random((model_config["n_flux"], model_config["n_latents"]))
        b_flux = rng.random((model_config["n_flux"], 1))
        A_label = rng.random((model_config["n_labels"], model_config["n_latents"]))

        pars = {
            "flux": {"data": [{"A": A_flux}, {"b": b_flux}]},
            "label": {"data": {"A": A_label}},
        }

        # Test packing
        packed = multi_output_model.pack_numpyro_pars(pars)
        expected_keys = {"flux:0:A", "flux:1:b", "label:A"}

        assert set(packed.keys()) == expected_keys
        assert np.allclose(packed["flux:0:A"], A_flux)
        assert np.allclose(packed["flux:1:b"], b_flux)
        assert np.allclose(packed["label:A"], A_label)

        # Test unpacking
        unpacked = multi_output_model.unpack_numpyro_pars(packed)

        # Check flux (TransformSequence - list structure)
        assert isinstance(unpacked["flux"]["data"], list | tuple)
        assert len(unpacked["flux"]["data"]) == 2
        assert np.allclose(unpacked["flux"]["data"][0]["A"], A_flux)
        assert np.allclose(unpacked["flux"]["data"][1]["b"], b_flux)

        # Check label (single transform - dict structure)
        assert isinstance(unpacked["label"]["data"], dict)
        assert np.allclose(unpacked["label"]["data"]["A"], A_label)

        # Check error parameters are empty but properly structured
        assert unpacked["flux"]["err"] == {}
        assert unpacked["label"]["err"] == {}

    def test_round_trip_conversion_integrity(
        self, transform_sequence_with_err_model, model_config, rng
    ):
        """Test that pack → unpack → pack preserves the original parameter structure.

        This test validates the mathematical integrity of the parameter conversion system:
        - Original parameter structure is perfectly preserved through round-trip conversion
        - No data is lost or corrupted during the conversion process
        - Floating-point precision is maintained within reasonable tolerances
        - The conversion process is mathematically invertible

        This is critical for ensuring that optimization results can be reliably
        converted between formats without introducing numerical errors.
        """
        # Create original parameters with both data and error components
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))
        b = rng.random((model_config["n_flux"], 1))
        err_b = rng.random((1,))  # Note: different parameter name but could conflict

        orig_pars = {
            "flux": {
                "data": [{"A": A}, {"b": b}],
                "err": {"scale": err_b},
            }
        }

        # Round trip: pack → unpack → pack
        packed = transform_sequence_with_err_model.pack_numpyro_pars(orig_pars)
        unpacked = transform_sequence_with_err_model.unpack_numpyro_pars(packed)
        repacked = transform_sequence_with_err_model.pack_numpyro_pars(unpacked)

        # Verify perfect round-trip conversion
        assert set(packed.keys()) == set(repacked.keys())
        for key in packed:
            assert np.allclose(packed[key], repacked[key])

    def test_missing_parameters_handling(
        self, single_transform_model, model_config, rng
    ):
        """Test graceful handling of missing parameters with ignore_missing flag.

        This test validates the robustness of the unpacking system when dealing with
        incomplete parameter sets:
        - With ignore_missing=True: missing parameters are gracefully ignored
        - With ignore_missing=False: missing parameters raise clear KeyError exceptions
        - Partial parameter sets are handled without corrupting existing data
        - Error messages are informative for debugging missing parameters

        This functionality is important for incremental parameter loading and
        debugging optimization issues.
        """
        # Create complete parameter set
        complete_packed = {
            "flux:A": rng.random((model_config["n_flux"], model_config["n_latents"]))
        }

        # Should handle complete parameters without issue
        unpacked = single_transform_model.unpack_numpyro_pars(
            complete_packed, ignore_missing=False
        )

        assert "flux" in unpacked
        assert "data" in unpacked["flux"]
        assert "A" in unpacked["flux"]["data"]
        assert unpacked["flux"]["err"] == {}

        # Test with truly empty parameters - this should work with ignore_missing
        empty_packed = {}
        unpacked_skipped = single_transform_model.unpack_numpyro_pars(
            empty_packed, ignore_missing=True
        )
        # With ignore_missing and no parameters, we get an empty dict (no outputs created)
        assert unpacked_skipped == {}


class TestLuxValidation:
    """Tests for input validation."""

    def test_output_name_with_colon_raises(self):
        """Output names containing colons should raise ValueError."""
        model = plx.LVM(latent_size=4)
        with pytest.raises(ValueError, match="contains ':'"):
            model.register_output("flux:invalid", LinearTransform(output_size=8))

    def test_direct_format_raises(self, rng, model_config):
        """Parameters must be nested under "data"/"err", as optimize() returns them."""
        model = plx.LVM(latent_size=model_config["n_latents"])
        model.register_output(
            "flux", LinearTransform(output_size=model_config["n_flux"])
        )

        latents = rng.random((model_config["n_stars"], model_config["n_latents"]))
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))

        with pytest.raises(ValueError, match="no 'data' or 'err' key"):
            model.predict_outputs({"flux": {"A": A}}, latents)

    def test_nested_format_no_warning(self, rng, model_config):
        """Using nested parameter format should not raise any warnings."""
        model = plx.LVM(latent_size=model_config["n_latents"])
        model.register_output(
            "flux", LinearTransform(output_size=model_config["n_flux"])
        )

        latents = rng.random((model_config["n_stars"], model_config["n_latents"]))
        A = rng.random((model_config["n_flux"], model_config["n_latents"]))

        # Nested format (correct) - should not warn
        nested_pars = {"flux": {"data": {"A": A}}}
        # If this raises a warning, pytest will fail due to filterwarnings=error
        model.predict_outputs(nested_pars, latents)

    def test_output_with_no_parameters_needs_no_entry(self, rng, model_config):
        """A NoOpTransform output never appears in a parameter dict, so don't demand it.

        unpack_numpyro_pars only emits entries for outputs that had sampled
        parameters, so requiring one here made every Cannon-style model (labels
        observed through a NoOpTransform) fail with KeyError.
        """
        n_latents, n_flux = model_config["n_latents"], model_config["n_flux"]
        model = plx.LVM(latent_size=n_latents)
        model.register_output("label", NoOpTransform())
        model.register_output("flux", LinearTransform(output_size=n_flux))

        latents = jnp.array(rng.random((model_config["n_stars"], n_latents)))
        A = jnp.array(rng.random((n_flux, n_latents)))

        # "label" is absent, and that must not be mistaken for a malformed entry
        out = model.predict_outputs({"flux": {"data": {"A": A}}}, latents)
        assert jnp.allclose(out["label"], latents)
        assert out["flux"].shape == (model_config["n_stars"], n_flux)


class TestPredictOutputsLatents:
    """predict_outputs takes pars first; the latents are optional."""

    @pytest.fixture
    def model_and_pars(self):
        rng = np.random.default_rng(9)
        model = plx.LVM(latent_size=2)
        model.register_output("flux", LinearTransform(output_size=3))
        pars = {
            "flux": {"data": {"A": jnp.array(rng.normal(size=(3, 2)))}},
            "latents": jnp.array(rng.normal(size=(5, 2))),
        }
        return model, pars

    def test_latents_read_from_pars(self, model_and_pars):
        """The common case: optimize() returns the latents inside pars."""
        model, pars = model_and_pars
        from_pars = model.predict_outputs(pars)
        explicit = model.predict_outputs(pars, pars["latents"])
        assert jnp.allclose(from_pars["flux"], explicit["flux"])

    def test_explicit_latents_win(self, model_and_pars):
        """Applying trained output parameters to a different set of objects."""
        model, pars = model_and_pars
        other = jnp.zeros((4, 2))
        out = model.predict_outputs(model.output_pars(pars), latents=other)
        assert out["flux"].shape == (4, 3)
        assert jnp.allclose(out["flux"], 0.0)

    def test_missing_latents_raises(self, model_and_pars):
        model, pars = model_and_pars
        with pytest.raises(KeyError, match="No latents given"):
            model.predict_outputs(model.output_pars(pars))

    def test_stale_argument_order_raises(self, model_and_pars):
        """The arguments used to be (latents, pars); catch calls that never moved."""
        model, pars = model_and_pars
        with pytest.raises(TypeError, match="takes the parameters"):
            model.predict_outputs(pars["latents"], pars)

    def test_output_pars_strips_the_latents(self, model_and_pars):
        model, pars = model_and_pars
        stripped = model.output_pars(pars)
        assert set(stripped) == {"flux"}
        assert jnp.array_equal(stripped["flux"]["data"]["A"], pars["flux"]["data"]["A"])
        assert "latents" in pars  # the original is untouched


class TestOutputParsPerObject:
    """output_pars promises the parameters that carry over to *other* objects.

    The latents are the obvious per-object parameters, but a transform can declare
    others via a "data_size" shape. Carrying one of those into a new dataset would
    either blow up on shape or, worse, silently apply one object's nuisance
    parameter to a different object.
    """

    @pytest.fixture
    def model_and_pars(self):
        n_data, n_out = 6, 4
        offset = FunctionTransform(
            output_size=n_out,
            transform=lambda y, offset: y + offset[:, None],
            priors={"offset": dist.Normal(0.0, 5.0)},
            shapes={"offset": ("data_size",)},
            vmap=False,
        )
        model = plx.LVM(latent_size=2)
        model.register_output(
            "flux",
            TransformSequence((LinearTransform(output_size=n_out), offset)),
        )
        pars = {
            "latents": jnp.zeros((n_data, 2)),
            "flux": {
                "data": ({"A": jnp.ones((n_out, 2))}, {"offset": jnp.arange(6.0)}),
                "err": {},
            },
        }
        return model, pars, n_data

    def test_per_object_names_found_by_measurement(self, model_and_pars):
        model, _, _ = model_and_pars
        assert model.per_object_param_names() == {"latents", "flux:1:offset"}

    def test_per_object_parameters_are_dropped(self, model_and_pars):
        model, pars, _ = model_and_pars
        stripped = model.output_pars(pars)

        assert "latents" not in stripped
        # The shared linear map survives; the per-object offset does not
        assert jnp.array_equal(
            stripped["flux"]["data"][0]["A"], pars["flux"]["data"][0]["A"]
        )
        assert "offset" not in stripped["flux"]["data"][1]

    def test_result_transfers_to_a_different_number_of_objects(self, model_and_pars):
        """The point of dropping them: the training-set size must not leak."""
        model, pars, n_data = model_and_pars
        other = jnp.zeros((n_data + 3, 2))  # deliberately a different size

        packed = model.pack_numpyro_pars(model.output_pars(pars), ignore_missing=True)
        assert not any(
            getattr(v, "shape", ()) and v.shape[0] == n_data for v in packed.values()
        )

    def test_shared_only_model_is_unaffected(self):
        """With no per-object transform parameters, only the latents go."""
        model = plx.LVM(latent_size=2)
        model.register_output("flux", LinearTransform(output_size=3))
        assert model.per_object_param_names() == {"latents"}


class TestOptimizeDefaults:
    """Default arguments of LVM.optimize that no test previously exercised."""

    @pytest.fixture
    def model_and_data(self):
        rng = np.random.default_rng(42)
        model = plx.LVM(latent_size=2)
        model.register_output("flux", LinearTransform(output_size=3))
        true_flux = rng.normal(size=(8, 2)) @ rng.normal(size=(3, 2)).T
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(true_flux, err=np.full_like(true_flux, 0.1))
        )
        return model, data

    def test_default_optimizer(self, model_and_data):
        """optimize() works without an explicit optimizer (numpyro Adam needs a step)."""
        model, data = model_and_data
        pars, _ = model.optimize(data, num_steps=2, rng_key=jax.random.PRNGKey(0))
        assert pars["latents"].shape == (8, 2)

    def test_improper_uniform_latents_prior(self, model_and_data):
        """latents_prior=False uses an improper uniform over the latents."""
        model, data = model_and_data
        pars, _ = model.optimize(
            data,
            num_steps=2,
            rng_key=jax.random.PRNGKey(0),
            optimizer=numpyro.optim.Adam(1e-3),
            latents_prior=False,
        )
        assert pars["latents"].shape == (8, 2)


class TestOptimizeGuide:
    """Tests for the guide parameter on LVM.optimize."""

    @pytest.fixture
    def model_and_data(self):
        """Create a simple model and synthetic data for optimize tests."""
        rng = np.random.default_rng(42)
        n_latents = 3
        n_flux = 6
        n_stars = 20

        model = plx.LVM(latent_size=n_latents)
        model.register_output("flux", LinearTransform(output_size=n_flux))

        true_A = rng.normal(size=(n_flux, n_latents)) * 0.5
        true_latents = rng.normal(size=(n_stars, n_latents))
        true_flux = true_latents @ true_A.T
        flux_err = np.full_like(true_flux, 0.1)

        data = plx.data.PolluxData(
            flux=plx.data.OutputData(
                true_flux + rng.normal(0, flux_err),
                err=flux_err,
            ),
        )
        return model, data

    def test_default_guide_is_autodelta(self, model_and_data):
        """Default guide (None) should use AutoDelta and return point estimates."""
        model, data = model_and_data
        key = jax.random.PRNGKey(0)

        pars, svi_results = model.optimize(
            data,
            num_steps=50,
            rng_key=key,
            optimizer=numpyro.optim.Adam(1e-3),
        )
        assert "latents" in pars
        assert "flux" in pars

    def test_guide_as_class(self, model_and_data):
        """Passing an AutoGuide subclass should work (instantiated internally)."""

        model, data = model_and_data
        key = jax.random.PRNGKey(1)

        pars, svi_results = model.optimize(
            data,
            num_steps=50,
            rng_key=key,
            optimizer=numpyro.optim.Adam(1e-3),
            guide=AutoNormal,
        )
        assert "latents" in pars
        assert "flux" in pars

    def test_guide_as_instance(self, model_and_data):
        """Passing a pre-constructed AutoGuide instance should work."""

        model, data = model_and_data
        key = jax.random.PRNGKey(2)

        # Build the model function the same way optimize does internally
        numpyro_model = partial(
            model.default_numpyro_model, latents_prior=None, custom_model=None
        )
        guide_instance = AutoNormal(numpyro_model)

        pars, svi_results = model.optimize(
            data,
            num_steps=50,
            rng_key=key,
            optimizer=numpyro.optim.Adam(1e-3),
            guide=guide_instance,
        )
        assert "latents" in pars
        assert "flux" in pars

    def test_invalid_guide_raises(self, model_and_data):
        """Passing an invalid guide type should raise TypeError."""

        model, data = model_and_data
        key = jax.random.PRNGKey(3)

        with pytest.raises(TypeError, match="guide must be"):
            model.optimize(
                data,
                num_steps=10,
                rng_key=key,
                optimizer=numpyro.optim.Adam(1e-3),
                guide="not_a_guide",
            )

    def test_autonormal_produces_different_params_than_autodelta(self, model_and_data):
        """AutoNormal and AutoDelta may produce different point estimates."""

        model, data = model_and_data

        pars_delta, _ = model.optimize(
            data,
            num_steps=100,
            rng_key=jax.random.PRNGKey(10),
            optimizer=numpyro.optim.Adam(1e-3),
        )
        pars_normal, _ = model.optimize(
            data,
            num_steps=100,
            rng_key=jax.random.PRNGKey(10),
            optimizer=numpyro.optim.Adam(1e-3),
            guide=AutoNormal,
        )

        # Both should have the same keys/structure
        assert set(pars_delta.keys()) == set(pars_normal.keys())
        assert set(pars_delta["flux"].keys()) == set(pars_normal["flux"].keys())

        # Latents should have the same shape
        assert pars_delta["latents"].shape == pars_normal["latents"].shape


class TestLatentUncertainties:
    """Tests for LVM.latent_uncertainties."""

    @pytest.fixture
    def linear(self):
        """A purely linear model, where the exact posterior covariance is known."""
        rng = np.random.default_rng(7)
        n_latents, n_flux, n_stars = 2, 8, 5

        model = plx.LVM(latent_size=n_latents)
        model.register_output("flux", LinearTransform(output_size=n_flux))

        A = jnp.asarray(rng.normal(size=(n_flux, n_latents)))
        latents = jnp.asarray(rng.normal(size=(n_stars, n_latents)))
        flux_err = jnp.asarray(rng.uniform(0.05, 0.2, size=(n_stars, n_flux)))
        flux = latents @ A.T + jnp.asarray(rng.normal(0, flux_err))

        data = plx.data.PolluxData(flux=plx.data.OutputData(flux, err=flux_err))
        pars = {"latents": latents, "flux": {"data": {"A": A}, "err": {}}}
        return model, data, pars, A, flux_err

    def test_matches_the_analytic_covariance(self, linear):
        """For a linear model the objective is quadratic, so this is exact."""
        model, data, pars, A, flux_err = linear

        cov = model.latent_uncertainties(data, pars, covariance=True)

        # (A^T W A + prior precision)^-1, star by star, with the default unit Gaussian
        for i in range(len(data)):
            W = jnp.diag(1.0 / flux_err[i] ** 2)
            expected = jnp.linalg.inv(A.T @ W @ A + jnp.eye(2))
            assert jnp.allclose(cov[i], expected, rtol=1e-5)

    def test_prior_tightens_the_uncertainties(self, linear):
        model, data, pars, _, _ = linear

        with_prior = model.latent_uncertainties(data, pars)
        without = model.latent_uncertainties(data, pars, latents_prior=False)
        tighter = model.latent_uncertainties(
            data, pars, latents_prior=dist.Normal(0, 0.1)
        )

        assert jnp.all(with_prior < without)
        assert jnp.all(tighter < with_prior)

    def test_sigma_is_the_covariance_diagonal(self, linear):
        model, data, pars, _, _ = linear

        sigma = model.latent_uncertainties(data, pars)
        cov = model.latent_uncertainties(data, pars, covariance=True)
        assert sigma.shape == (len(data), model.latent_size)
        assert jnp.allclose(sigma, jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1)))

    def test_larger_errors_give_larger_uncertainties(self, linear):
        model, data, pars, _, flux_err = linear
        noisier = plx.data.PolluxData(
            flux=plx.data.OutputData(data["flux"].data, err=flux_err * 5)
        )
        assert jnp.all(
            model.latent_uncertainties(noisier, pars)
            > model.latent_uncertainties(data, pars)
        )

    def test_latents_can_be_passed_explicitly(self, linear):
        model, data, pars, _, _ = linear
        sigma = model.latent_uncertainties(data, pars, latents=pars["latents"])
        assert jnp.allclose(sigma, model.latent_uncertainties(data, pars))

    def test_missing_latents_raises(self, linear):
        model, data, pars, _, _ = linear
        no_latents = {k: v for k, v in pars.items() if k != "latents"}
        with pytest.raises(KeyError, match="No latents given"):
            model.latent_uncertainties(data, no_latents)

    def test_output_without_data_raises(self, linear):
        model, data, pars, _, _ = linear
        with pytest.raises(ValueError, match="No data for output"):
            model.latent_uncertainties(data, pars, names="label")


class TestOptimizeInitLocFn:
    """Tests for the init_loc_fn parameter on LVM.optimize."""

    @pytest.fixture
    def model_and_data(self):
        """A tiny linear model and some data to optimize it against."""
        rng = np.random.default_rng(42)
        model = plx.LVM(latent_size=3)
        model.register_output("flux", LinearTransform(output_size=6))

        flux = rng.normal(size=(20, 6))
        data = plx.data.PolluxData(
            flux=plx.data.OutputData(flux, err=np.full_like(flux, 0.1)),
        )
        return model, data

    def test_starts_from_the_given_values(self, model_and_data):
        """One step at a negligible step size should barely move off the start."""
        model, data = model_and_data

        start = {
            "latents": jnp.full((20, 3), 0.3),
            "flux": {"data": {"A": jnp.full((6, 3), 0.7)}},
        }
        pars, _ = model.optimize(
            data,
            num_steps=1,
            rng_key=jax.random.PRNGKey(0),
            optimizer=numpyro.optim.Adam(1e-8),
            init_loc_fn=init_to_value(values=model.pack_numpyro_pars(start)),
        )

        assert np.allclose(pars["latents"], 0.3, atol=1e-5)
        assert np.allclose(pars["flux"]["data"]["A"], 0.7, atol=1e-5)

    def test_partial_values_fall_back_to_the_default(self, model_and_data):
        """Sites left out of the values dict get the guide's own initialization."""
        model, data = model_and_data

        start = {"flux": {"data": {"A": jnp.full((6, 3), 0.7)}}}
        pars, _ = model.optimize(
            data,
            num_steps=1,
            rng_key=jax.random.PRNGKey(0),
            optimizer=numpyro.optim.Adam(1e-8),
            init_loc_fn=init_to_value(
                values=model.pack_numpyro_pars(start, ignore_missing=True)
            ),
        )

        assert np.allclose(pars["flux"]["data"]["A"], 0.7, atol=1e-5)
        assert pars["latents"].shape == (20, 3)

    def test_guide_instance_raises(self, model_and_data):
        """An already-constructed guide has picked its own starting point."""
        model, data = model_and_data

        numpyro_model = partial(
            model.default_numpyro_model, latents_prior=None, custom_model=None
        )

        with pytest.raises(ValueError, match="init_loc_fn has no effect"):
            model.optimize(
                data,
                num_steps=1,
                rng_key=jax.random.PRNGKey(0),
                guide=AutoNormal(numpyro_model),
                init_loc_fn=init_to_value(values={}),
            )
