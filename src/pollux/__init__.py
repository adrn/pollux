"""
Copyright (c) 2024-2025 adrn. All rights reserved.

pollux: Data-driven models of stellar spectra.
"""

from __future__ import annotations

from ._src import data, models
from ._version import version as __version__

__all__ = ["__version__", "data", "models"]
