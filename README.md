<img src="https://pollux-astro.readthedocs.io/en/latest/_static/Pollux-logo.png" alt="Pollux logo" width="200"/>

Pollux is a framework for building data-driven latent variable models in JAX: you define
a latent space and compose transforms from it to any number of observed outputs. It
ships with _Lux_ and the _Cannon_ as ready-made models for stellar spectroscopy, but the
model-building pieces are general.

[![Documentation Status][rtd-badge]][rtd-link]
[![Actions Status][actions-badge]][actions-link]
[![Coverage Status][codecov-badge]][codecov-link]

<!-- [![PyPI version][pypi-version]][pypi-link] -->
<!-- [![PyPI platforms][pypi-platforms]][pypi-link] -->

<!-- SPHINX-START -->

## Installation

`pollux` requires Python 3.12 or newer:

```bash
pip install pollux    # or: uv add pollux
```

For the unreleased development version, install from this repository:

```bash
pip install git+https://github.com/adrn/pollux
```

## Development

```bash
uv sync                    # all development dependencies, in .venv
uv run pytest              # run the tests
```

The [PEP 735](https://peps.python.org/pep-0735/) dependency groups (`test`, `docs`,
`dev`) also work with `pip` 25.1 or newer:

```bash
python -m pip install -e . --group dev
```

See [DEV.md](DEV.md) for building the docs and cutting a release, and the
[documentation][rtd-link] for everything else.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/adrn/pollux/workflows/CI/badge.svg
[actions-link]:             https://github.com/adrn/pollux/actions
[codecov-badge]:            https://codecov.io/gh/adrn/pollux/graph/badge.svg?token=54TQPUSI2F
[codecov-link]:             https://codecov.io/gh/adrn/pollux
[pypi-link]:                https://pypi.org/project/pollux/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/pollux
[pypi-version]:             https://img.shields.io/pypi/v/pollux
[rtd-badge]:                https://readthedocs.org/projects/pollux-astro/badge/?version=latest
[rtd-link]:                 https://pollux-astro.readthedocs.io/en/latest/?badge=latest
<!-- prettier-ignore-end -->
