"""Tests for the Lux architecture."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import pollux as plx
from pollux.models.lvm import select_outputs
from pollux.models.transforms import (
    LinearTransform,
    NoOpTransform,
    QuadraticTransform,
    ScatterTransform,
)


class TestSelectOutputs:
    """The selector shared by the architectures for their per-output options."""

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (True, {"label": None, "flux": None}),
            (False, {}),
            (None, {}),
            ([], {}),
            (["flux"], {"flux": None}),
            (("label", "flux"), {"label": None, "flux": None}),
            ({"flux": 5.0}, {"flux": 5.0}),
            # A bare name is a Sequence[str] of its own characters, so this would
            # otherwise select 'f', 'l', 'u', 'x'
            ("flux", {"flux": None}),
        ],
    )
    def test_forms(self, spec, expected):
        assert select_outputs(spec, ["label", "flux"], "scatter") == expected

    def test_unknown_name_raises(self):
        """A typo in a selector should not silently do nothing."""
        with pytest.raises(ValueError, match=r"\['flx'\] are not outputs"):
            select_outputs(["flx"], ["label", "flux"], "intrinsic_scatter")

    def test_unknown_bare_name_raises(self):
        """A mistyped bare name reports itself, not its letters."""
        with pytest.raises(ValueError, match=r"\['flx'\] are not outputs"):
            select_outputs("flx", ["label", "flux"], "intrinsic_scatter")


class TestLuxConstruction:
    def test_registers_one_output_per_entry(self):
        model = plx.Lux(latent_size=16, outputs={"label": 6, "flux": 1000})
        assert list(model.outputs) == ["label", "flux"]
        assert model.latent_size == 16
        assert model.outputs["label"].data_transform.output_size == 6
        assert model.outputs["flux"].data_transform.output_size == 1000

    def test_is_an_lvm(self):
        model = plx.Lux(latent_size=4, outputs={"flux": 8})
        assert isinstance(model, plx.LVM)

    def test_linear_by_default_quadratic_on_request(self):
        model = plx.Lux(
            latent_size=4, outputs={"label": 3, "flux": 8}, quadratic=["flux"]
        )
        assert isinstance(model.outputs["label"].data_transform, LinearTransform)
        assert isinstance(model.outputs["flux"].data_transform, QuadraticTransform)

    def test_quadratic_true_selects_every_output(self):
        model = plx.Lux(latent_size=4, outputs={"label": 3, "flux": 8}, quadratic=True)
        for output in model.outputs.values():
            assert isinstance(output.data_transform, QuadraticTransform)

    def test_scatter_on_by_default(self):
        model = plx.Lux(latent_size=4, outputs={"flux": 8})
        assert isinstance(model.outputs["flux"].err_transform, ScatterTransform)

    def test_scatter_can_be_turned_off(self):
        model = plx.Lux(latent_size=4, outputs={"flux": 8}, intrinsic_scatter=False)
        assert isinstance(model.outputs["flux"].err_transform, NoOpTransform)

    def test_scatter_selected_per_output(self):
        model = plx.Lux(
            latent_size=4,
            outputs={"label": 3, "flux": 8},
            intrinsic_scatter=["flux"],
        )
        assert isinstance(model.outputs["flux"].err_transform, ScatterTransform)
        assert isinstance(model.outputs["label"].err_transform, NoOpTransform)

    def test_scatter_prior_scale_from_mapping(self):
        """The scale that suits an output depends on its preprocessing."""
        model = plx.Lux(
            latent_size=4,
            outputs={"label": 3, "flux": 8},
            intrinsic_scatter={"flux": 5.0},
        )
        prior = model.outputs["flux"].err_transform.priors["s"]
        assert isinstance(prior, dist.HalfNormal)
        assert np.isclose(prior.scale, 5.0)
        assert isinstance(model.outputs["label"].err_transform, NoOpTransform)

    def test_default_scatter_prior_scale_is_one(self):
        model = plx.Lux(latent_size=4, outputs={"flux": 8})
        assert np.isclose(model.outputs["flux"].err_transform.priors["s"].scale, 1.0)

    def test_no_outputs_raises(self):
        """An architecture with no outputs is a mistake, not an empty framework model."""
        with pytest.raises(ValueError, match="at least one output"):
            plx.Lux(latent_size=4, outputs={})

    def test_unknown_selector_name_raises(self):
        with pytest.raises(ValueError, match="not outputs of this model"):
            plx.Lux(latent_size=4, outputs={"flux": 8}, quadratic=["spectrum"])

    def test_selectors_accept_a_bare_output_name(self):
        """quadratic="flux" should mean the flux, not the letters f, l, u and x."""
        model = plx.Lux(
            latent_size=4,
            outputs={"label": 3, "flux": 8},
            quadratic="flux",
            intrinsic_scatter="flux",
        )
        assert isinstance(model.outputs["flux"].data_transform, QuadraticTransform)
        assert isinstance(model.outputs["label"].data_transform, LinearTransform)
        assert isinstance(model.outputs["flux"].err_transform, ScatterTransform)
        assert isinstance(model.outputs["label"].err_transform, NoOpTransform)


class TestLuxMatchesHandBuiltModel:
    """The sugar has to be faithful to the model it replaces.

    This is the structure the APOGEE tutorials build by hand: labels linear in the
    latents with their reported errors taken at face value, flux linear in the latents
    with a fitted per-pixel scatter.
    """

    @pytest.fixture
    def sizes(self):
        return {"n_stars": 24, "n_latents": 3, "n_labels": 4, "n_flux": 12}

    @pytest.fixture
    def hand_built(self, sizes):
        model = plx.LVM(latent_size=sizes["n_latents"])
        model.register_output("label", LinearTransform(output_size=sizes["n_labels"]))
        model.register_output(
            "flux",
            LinearTransform(output_size=sizes["n_flux"]),
            err_transform=ScatterTransform(
                output_size=sizes["n_flux"], priors={"s": dist.HalfNormal(5.0)}
            ),
        )
        return model

    @pytest.fixture
    def sugared(self, sizes):
        return plx.Lux(
            latent_size=sizes["n_latents"],
            outputs={"label": sizes["n_labels"], "flux": sizes["n_flux"]},
            intrinsic_scatter={"flux": 5.0},
        )

    @pytest.fixture
    def data(self, sizes):
        rng = np.random.default_rng(7)
        latents = rng.normal(size=(sizes["n_stars"], sizes["n_latents"]))
        label = latents @ rng.normal(size=(sizes["n_labels"], sizes["n_latents"])).T
        flux = latents @ rng.normal(size=(sizes["n_flux"], sizes["n_latents"])).T
        return plx.data.PolluxData(
            label=plx.data.OutputData(jnp.array(label), err=jnp.full(label.shape, 0.1)),
            flux=plx.data.OutputData(jnp.array(flux), err=jnp.full(flux.shape, 0.05)),
        )

    def test_same_priors(self, hand_built, sugared):
        for name in hand_built.outputs:
            hand, sug = hand_built.outputs[name], sugared.outputs[name]
            for kind in ("data_transform", "err_transform"):
                hand_priors = getattr(hand, kind).get_expanded_priors(
                    latent_size=hand_built.latent_size, data_size=24
                )
                sug_priors = getattr(sug, kind).get_expanded_priors(
                    latent_size=sugared.latent_size, data_size=24
                )
                assert hand_priors.keys() == sug_priors.keys()
                for key, prior in hand_priors.items():
                    assert type(prior) is type(sug_priors[key])
                    assert prior.batch_shape == sug_priors[key].batch_shape

    def test_same_optimized_loss(self, hand_built, sugared, data):
        """Identical structure and identical seed must give an identical fit."""
        kwargs = {
            "max_cycles": 3,
            "rng_key": jax.random.PRNGKey(0),
            "progress": False,
        }
        hand_res = hand_built.optimize_iterative(data, **kwargs)
        sug_res = sugared.optimize_iterative(data, **kwargs)

        assert [b.name for b in hand_res.blocks] == [b.name for b in sug_res.blocks]
        assert np.allclose(
            hand_res.losses_per_cycle, sug_res.losses_per_cycle, rtol=1e-10
        )

    def test_predicts_through_the_framework(self, sugared, data, sizes):
        res = sugared.optimize_iterative(
            data, max_cycles=2, rng_key=jax.random.PRNGKey(0), progress=False
        )
        pred = sugared.predict_outputs(res.params)
        assert pred["label"].shape == (sizes["n_stars"], sizes["n_labels"])
        assert pred["flux"].shape == (sizes["n_stars"], sizes["n_flux"])
