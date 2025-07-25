"""BNL: A Python library for hierarchical text segmentation and evaluation.

This library provides tools for working with hierarchical text segments.

Submodules
----------
data
    Data loading and management for musical structure datasets.
ops
    Operations and transformations on BNL objects.
viz
    Visualization tools for BNL objects.
"""

from __future__ import annotations

__version__ = "0.0.3"

# --- Import submodules for explicit, namespaced access ---
from . import core, data, ops, viz

# --- Promote the core data structures to the top level ---
from .core import (
    Boundary as B,
    BoundaryContour as BC,
    BoundaryHierarchy as BH,
    LeveledBoundary as LB,
    MultiSegment as MS,
    RatedBoundary as RB,
    Segment as S,
    TimeSpan as TS,
)

# --- Define the public-facing API of the `bnl` package ---
__all__ = [
    # Point
    "B",
    "RB",
    "LB",
    # Containers
    "TS",
    "S",
    "MS",
    # Monotonic Boundaries
    "BC",
    "BH",
    # Submodules (toolboxes)
    "data",
    "ops",
    "viz",
    "core",
]
