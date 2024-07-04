import jax
import flax.linen as nn


class Paton(nn.Module):
    n_labels: int
    n_latents: int
    n_dense: int
    n_flux: int

    def setup(self):
        self.labels = nn.Dense(self.n_labels)
        self.hidden = nn.Dense(self.n_dense)
        self.hidden2 = nn.Dense(self.n_dense)
        self.flux = nn.Dense(self.n_flux)

    def __call__(self, z):
        labels = self.labels(z)

        interim = jax.nn.sigmoid(self.hidden(z))
        interim2 = jax.nn.sigmoid(self.hidden2(interim))
        flux = jax.nn.sigmoid(self.flux(interim2))

        return labels, flux


class Paton:

    def __init__(self, label_data, spectrum_data, *args, **kwargs):
        pass