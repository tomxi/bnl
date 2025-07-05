"""BNL: A Python library for hierarchical text segmentation and evaluation.

This library provides tools for working with hierarchical text segments.

Submodules
----------
core
    Core data structures and functionality.
data
    Data loading and management for musical structure datasets.
ops
    Operations and transformations on BNL objects.
"""

from __future__ import annotations

__version__ = "0.2.1"

# --- Import submodules for explicit, namespaced access ---
from . import core, data, ops, strategies

# --- Promote the core data structures to the top level ---
from .core import (
    Boundary,
    Hierarchy,
    ProperHierarchy,
    RatedBoundary,
    Segmentation,
    TimeSpan,
)
from .ops import Pipeline

# --- Define the public-facing API of the `bnl` package ---
__all__ = [
    # Promoted from .core
    "Boundary",
    "TimeSpan",
    "Segmentation",
    "Hierarchy",
    "ProperHierarchy",
    "RatedBoundary",
    # Promoted from .ops
    "Pipeline",
    # Submodules (toolboxes)
    "core",
    "data",
    "ops",
    "strategies",
]
