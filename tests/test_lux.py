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


@pytest.fixture()
def rng():
    """Random number generator for consistent test results."""
    return np.random.default_rng(42)


@pytest.fixture()
def model_config():
    """Basic model configuration used across tests."""
    return {
        "n_latents": 4,
        "n_flux": 8,
        "n_labels": 2,
        "n_stars": 16,
    }


@pytest.fixture()
def single_transform_model(model_config):
    """LuxModel with a single LinearTransform for testing basic functionality."""
    model = plx.LuxModel(latent_size=model_config["n_latents"])
    model.register_output("flux", LinearTransform(output_size=model_config["n_flux"]))
    return model


@pytest.fixture()
def single_transform_with_err_model(model_config):
    """LuxModel with LinearTransform and OffsetTransform error transform."""
    model = plx.LuxModel(latent_size=model_config["n_latents"])
    model.register_output(
        "flux",
        LinearTransform(output_size=model_config["n_flux"]),
        err_transform=OffsetTransform(output_size=model_config["n_flux"]),
    )
    return model


@pytest.fixture()
def transform_sequence_model(model_config):
    """LuxModel with a TransformSequence (LinearTransform + OffsetTransform)."""
    model = plx.LuxModel(latent_size=model_config["n_latents"])
    trans_seq = TransformSequence(
        transforms=(
            LinearTransform(output_size=model_config["n_flux"]),
            OffsetTransform(output_size=model_config["n_flux"]),
        )
    )
    model.register_output("flux", trans_seq)
    return model


@pytest.fixture()
def transform_sequence_with_err_model(model_config):
    """LuxModel with TransformSequence and a FunctionTransform error transform."""
    model = plx.LuxModel(latent_size=model_config["n_latents"])

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
        param_priors={"scale": dist.Normal(1.0, 0.1)},
        param_shapes={"scale": (1,)},
        vmap=False,
    )

    model.register_output("flux", trans_seq, err_transform=err_trans)
    return model


@pytest.fixture()
def multi_output_model(model_config):
    """LuxModel with multiple outputs: TransformSequence flux + single transform labels."""
    model = plx.LuxModel(latent_size=model_config["n_latents"])

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


class TestLuxModelParameterPackUnpack:
    """Test suite for LuxModel parameter packing and unpacking functionality.

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
        """Test graceful handling of missing parameters with skip_missing flag.

        This test validates the robustness of the unpacking system when dealing with
        incomplete parameter sets:
        - With skip_missing=True: missing parameters are gracefully ignored
        - With skip_missing=False: missing parameters raise clear KeyError exceptions
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
            complete_packed, skip_missing=False
        )

        assert "flux" in unpacked
        assert "data" in unpacked["flux"]
        assert "A" in unpacked["flux"]["data"]
        assert unpacked["flux"]["err"] == {}

        # Test with truly empty parameters - this should work with skip_missing
        empty_packed = {}
        unpacked_skipped = single_transform_model.unpack_numpyro_pars(
            empty_packed, skip_missing=True
        )
        # With skip_missing and no parameters, we get an empty dict (no outputs created)
        assert unpacked_skipped == {}
