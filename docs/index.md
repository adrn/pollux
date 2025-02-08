# pollux

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials

tutorials/Simulated-data-test.ipynb
```

## Get Started

Pollux is a Python library for constructing generative models of astronomical spectra
and other kinds of data. It is built on top of [JAX][jax] and is designed for use in
probabilistic and machine learning contexts.

Two classes of models are currently supported:

- [_Lux_](https://arxiv.org/abs/2502.01745): Multi-output, generative, latent variable
  models for inferring embedded representations of spectroscopic and many other kinds of
  data.
- [_Cannon_](https://arxiv.org/abs/1501.07604): Data-driven models for inferring stellar
  parameters, element abundances, and other labels from stellar spectra.

---

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[jax]: https://jax.readthedocs.io/en/latest/
[pypi-link]: https://pypi.org/project/TODO/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/TODO
[pypi-version]: https://img.shields.io/pypi/v/TODO
