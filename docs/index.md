# pollux

## Introduction

Pollux is a Python library for building data-driven latent variable models. You define a
latent space and compose transforms from it to any number of observed outputs, which can
be heterogeneous — spectra, photometry, stellar labels, or anything else measured for
the same objects. Pollux keeps track of the model parameters and hands the assembled
model to [numpyro][numpyro] for inference. It is built on [JAX][jax] and is designed for
use in probabilistic and machine learning contexts.

The framework is general, but it was written with stellar spectroscopy in mind, and two
ready-made models ship with it:

- [_Lux_](https://arxiv.org/abs/2502.01745): Multi-output, generative, latent variable
  models for inferring embedded representations of spectroscopic and many other kinds of
  data.
- [_Cannon_](https://arxiv.org/abs/1501.07604): Data-driven models for inferring stellar
  parameters, element abundances, and other labels from stellar spectra.

---

## Installation

`pollux` requires Python 3.12 or newer. Install the latest release with `pip`:

```bash
pip install pollux
```

or with [uv][uv]:

```bash
uv add pollux
```

To install the unreleased development version, point either tool at the repository
instead:

```bash
pip install git+https://github.com/adrn/pollux
uv add git+https://github.com/adrn/pollux
```

### Development installation

Clone the repository, then set up an environment with the development dependencies — the
test suite, the documentation build, and the linters:

```bash
uv sync                                        # everything, in .venv
uv sync --no-default-groups --group test       # just the test dependencies
```

(`uv sync --only-group test` would install the test tools _without_ `pollux` or its
runtime dependencies, which is rarely what you want.)

The same dependency groups work with `pip` 25.1 or newer, which added support for [PEP
735][pep735] groups:

```bash
python -m pip install -U pip
python -m pip install -e . --group dev
python -m pip install -e . --group test
```

Then run the tests with `uv run pytest` or `pytest`.

[jax]: https://jax.readthedocs.io/en/latest/
[numpyro]: https://num.pyro.ai/
[uv]: https://docs.astral.sh/uv/
[pep735]: https://peps.python.org/pep-0735/

## Get Started

The best way to get started with `pollux` is to work through the tutorials:

```{eval-rst}
.. include:: _tutorials.rst
```

```{toctree}
:maxdepth: 1
:caption: Technical Notes

linear-solves.md
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: API Reference

api/index.md
```
