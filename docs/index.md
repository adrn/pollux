# pollux

## Introduction

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
[uv]: https://docs.astral.sh/uv/
[pep735]: https://peps.python.org/pep-0735/

## Get Started

The best way to get started with `pollux` is to work through the tutorials:

```{eval-rst}
.. include:: _tutorials.rst
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: API Reference

api/index.md
```
