from __future__ import annotations

import importlib.metadata

import pollux as m


def test_version():
    assert importlib.metadata.version("pollux") == m.__version__
