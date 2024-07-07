from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module


class PatonNN(nn.Module):
    n_labels: int
    n_latents: int
    n_hidden: int | tuple[int]
    n_flux: int

    def setup(self) -> None:
        self.labels = nn.Dense(self.n_labels)
        if hasattr(self.n_hidden, "__iter__"):
            hidden_layers = []
            for i, n in enumerate(self.n_hidden):
                setattr(self, f"hidden_{i}", nn.Dense(n))
                hidden_layers.append(getattr(self, f"hidden_{i}"))
        else:
            self.hidden = nn.Dense(self.n_hidden)
            hidden_layers = [self.hidden]

        self._hidden_layers = hidden_layers
        self.flux = nn.Dense(self.n_flux)

    def __call__(self, z: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Linear transformation of the latents:
        labels = self.labels(z)

        # potentially "deep" layers
        hidden = z
        for layer in self._hidden_layers:
            # TODO: relu or sigmoid? APW thinks sigmoid
            hidden = jax.nn.sigmoid(layer(hidden))

        # Assumes the flux is continuum normalized to [0,1]ish
        flux = self.flux(hidden)

        return labels, flux

    def numpyro_model(self, data: dict[str, jax.Array]) -> None:
        n_stars = data["label"].shape[0]
        assert data["flux"].shape[0] == n_stars

        paton = random_flax_module(
            "paton",
            self,
            prior=dist.Normal(),
            # Note: could instead specify custom priors for the kernel/bias of each layer
            # prior={
            #     "flux.bias": dist.Normal(scale=0.1),
            #     "flux.kernel": dist.Normal(),
            #     ...
            input_shape=(n_stars, self.n_latents),
        )

        mean = numpyro.sample(
            "zeta_mean", dist.Normal(), sample_shape=(self.n_latents,)
        )
        log_std = numpyro.sample(
            "zeta_logstd", dist.Normal(), sample_shape=(self.n_latents,)
        )
        zeta = numpyro.sample(
            "zeta",
            dist.Normal(mean, jnp.exp(log_std)),
            sample_shape=(n_stars,),
        )

        # TODO: additional L1 regularization on the latents beyond n_labels?
        # numpyro.factor("zeta_L1", -jnp.abs(zeta[:, self.n_labels:] / 0.1).sum())

        model_label, model_flux = paton(zeta)
        numpyro.sample(
            "label", dist.Normal(model_label, data["label_err"]), obs=data["label"]
        )
        numpyro.sample(
            "flux", dist.Normal(model_flux, data["flux_err"]), obs=data["flux"]
        )

    def numpyro_model_label(
        self,
        data: dict[str, jax.Array],
        paton_params: dict[str, Any],
        other_params: dict[str, jax.Array],
    ) -> None:
        n_stars = data["label"].shape[0]
        zeta = numpyro.sample(
            "zeta",
            dist.Normal(
                other_params["zeta_mean"], jnp.exp(other_params["zeta_logstd"])
            ),
            sample_shape=(n_stars,),
        )

        model_label, model_flux = self.apply(paton_params, zeta)
        numpyro.sample(
            "flux", dist.Normal(model_flux, data["flux_err"]), obs=data["flux"]
        )
        numpyro.deterministic("label", model_label)

    def numpyro_model_flux(
        self,
        data: dict[str, jax.Array],
        paton_params: dict[str, Any],
        other_params: dict[str, jax.Array],
    ) -> None:
        n_stars = data["label"].shape[0]
        zeta = numpyro.sample(
            "zeta",
            dist.Normal(
                other_params["zeta_mean"], jnp.exp(other_params["zeta_logstd"])
            ),
            sample_shape=(n_stars,),
        )

        model_label, model_flux = self.apply(paton_params, zeta)
        numpyro.sample(
            "label", dist.Normal(model_label, data["label_err"]), obs=data["label"]
        )
        numpyro.deterministic("flux", model_flux)
