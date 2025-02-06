"""
Copyright (c) 2024-2025 adrn. All rights reserved.

pollux: Data-driven models of stellar spectra.
"""

from __future__ import annotations

from . import data, models
from ._version import version as __version__
from .models import LuxModel

__all__ = ["LuxModel", "__version__", "data", "models"]
