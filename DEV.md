# Development

## Setup

```bash
uv sync              # all dev dependencies (tests, docs, linters) into .venv
pre-commit install   # once, so the linters run on commit
```

With pip 25.1 or newer instead of uv: `python -m pip install -e . --group dev`.

## Tests

```bash
uv run pytest                            # everything (~70s)
uv run pytest --ignore=tests/integration # skip the slow SVI fits (~14s)
```

`pytest` collects `src`, `docs`, and `tests`, so docstring examples run too.
`tests/integration/` fits real models and accounts for most of the runtime.

Type checking and linters, on all files rather than just staged ones:

```bash
uv run ty check src
uv run pre-commit run --all-files   # includes ty, ruff, and the rest
```

## Docs

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. Notebooks are **not** executed by the build
(`nb_execution_mode = "off"` in `docs/conf.py`); their stored output is rendered as-is.
To re-run one:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace docs/tutorials/<name>.ipynb
```

The APOGEE tutorials need `docs/_data/rgb-highSNR-1k-1chip.h5`, which the `tutorials`
workflow downloads from <https://users.flatironinstitute.org/~apricewhelan/pollux/>. On
Read the Docs the notebooks come from that workflow's artifact, not from a local run.

## CI

`CI` runs pre-commit, the test matrix, and a Sphinx build; `Tutorials` executes the
notebooks. On a pull request each can be turned on or off:

| Job                   | PR default  | How to change it                                            |
| --------------------- | ----------- | ----------------------------------------------------------- |
| Tests                 | runs        | `skip tests` label, or `[skip tests]` in the commit subject |
| Docs (Sphinx build)   | runs        | `skip docs` label, or `[skip docs]` in the commit subject   |
| Tutorials (notebooks) | **skipped** | add the `tutorials` label to opt in                         |

The commit-subject flags are read from the branch head, not the PR merge commit. All of
this is PR-only: pushes to `main` and published releases always run everything, so
`[skip tests]` on a commit that lands on `main` has no effect.

Notebook execution is opt-in because it is the slowest job in the repo — and because
Read the Docs pulls the executed notebooks from that workflow's artifact, as described
above. The `Docs` job never executes notebooks, so it only checks that the docs build.

## Release

The version comes from the git tag via `hatch-vcs`, so there is no version number to
edit anywhere.

1. Make sure `main` is green.
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`
3. Publish a GitHub Release for that tag.

The `Release` workflow then builds the sdist and wheel and uploads them to PyPI with
Trusted Publishing. To check the artifacts first, without releasing:

```bash
uv build && uv run --with twine twine check dist/*
```

**Before the first release**, the `pollux` project must exist on PyPI with a Trusted
Publisher configured for owner `adrn`, repository `pollux`, workflow `release.yml`, and
environment `pypi` — otherwise the publish step fails.
