"""
Copyright (c) 2024-2025 adrn. All rights reserved.

pollux: Data-driven latent variable models in JAX.
"""

from __future__ import annotations

from . import data, models
from ._version import version as __version__
from .models import LVM, Cannon, Lux

__all__ = [
    "LVM",
    "Cannon",
    "Lux",
    "__version__",
    "data",
    "models",
]
