from __future__ import annotations

import importlib.metadata

import paton_nn as m


def test_version():
    assert importlib.metadata.version("paton_nn") == m.__version__
