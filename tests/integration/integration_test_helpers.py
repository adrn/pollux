import numpy as np
from jax.typing import ArrayLike


def make_simulated_linear_data(
    n_stars: int,
    n_latents: int,
    n_labels: int,
    n_flux: int,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, ArrayLike], dict[str, ArrayLike]]:
    rng = rng or np.random.default_rng()
    rngs = rng.spawn(4)

    if A is None:
        A = rngs[0].normal(size=(n_labels, n_latents))

    if B is None:
        B = rngs[1].normal(size=(n_flux, n_latents))

    assert A.shape == (n_labels, n_latents)
    assert B.shape == (n_flux, n_latents)

    latents = rng.normal(size=(n_stars, n_latents))

    with np.errstate(all="ignore"):
        labels = (A @ latents.T).T
        fluxs = (B @ latents.T).T
    assert np.all(np.isfinite(labels))
    assert np.all(np.isfinite(fluxs))

    label_err = rng.uniform(0.01, 0.1, size=labels.shape)
    flux_err = rng.uniform(0.01, 0.1, size=fluxs.shape)

    obs = {
        "label": rng.normal(labels, label_err),
        "label_err": label_err,
        "flux": rng.normal(fluxs, flux_err),
        "flux_err": flux_err,
    }
    truth = {
        "label": labels,
        "flux": fluxs,
        "A": A,
        "B": B,
        "latents": latents,
    }

    return obs, truth
