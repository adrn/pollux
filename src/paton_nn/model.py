from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module


class PatonNN(nn.Module):
    n_label: int
    n_latent: int
    n_hidden: int | tuple[int]
    n_flux: int

    def setup(self) -> None:
        self.label = nn.Dense(self.n_label)
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
        label = self.label(z)

        # potentially "deep" layers
        hidden = z
        for layer in self._hidden_layers:
            # TODO: relu or sigmoid or what?
            # hidden = jax.nn.sigmoid(layer(hidden))
            hidden = jax.nn.softplus(layer(hidden))

        # Assumes the flux is continuum normalized to [0,1]ish
        flux = self.flux(hidden)

        return label, flux

    def numpyro_model(self, data: dict[str, jax.Array]) -> None:
        n_stars = data["label"].shape[0]
        assert data["flux"].shape[0] == n_stars

        label_k_scale = jnp.concatenate(
            (jnp.ones(self.n_label), 0.1 * jnp.ones(self.n_latent - self.n_label))
        )[:, None]
        # label_k_scale = 0.1
        flax_priors = {
            "flux.bias": dist.Normal(),
            "flux.kernel": dist.Laplace(scale=0.1),
            "label.bias": dist.Normal(),
            "label.kernel": dist.Laplace(scale=label_k_scale),
        }
        if isinstance(self.n_hidden, int):
            flax_priors["hidden.bias"] = dist.Normal()
            flax_priors["hidden.kernel"] = dist.Laplace()
        else:
            for i, _ in enumerate(self.n_hidden):
                flax_priors[f"hidden_{i}.bias"] = dist.Normal()
                flax_priors[f"hidden_{i}.kernel"] = dist.Laplace()

        # flax_priors = dist.Normal()

        paton = random_flax_module(
            "paton",
            self,
            prior=flax_priors,
            input_shape=(n_stars, self.n_latent),
        )

        mean = numpyro.sample("zeta_mean", dist.Normal(), sample_shape=(self.n_latent,))

        # TODO: revisit this prior?
        log_std = numpyro.sample(
            "zeta_logstd", dist.Normal(-2, 1), sample_shape=(self.n_latent,)
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
            "label_obs", dist.Normal(model_label, data["label_err"]), obs=data["label"]
        )
        numpyro.sample(
            "flux_obs", dist.Normal(model_flux, data["flux_err"]), obs=data["flux"]
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
