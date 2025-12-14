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
from . import core, data, ops, viz, metrics, exp, lsd  # noqa: I001

from . import relevance as rel

# --- Promote the core data structures to the top level ---
from .core import (
    Boundary as B,
    LeveledBoundary as LB,
    RatedBoundary as RB,
    TimeSpan as TS,
    Segment as S,
    MultiSegment as MS,
    BoundaryContour as BC,
    BoundaryHierarchy as BH,
    LabelAgreementMap as LAM,
    SegmentAgreementProb as SAP,
    SegmentAffinityMatrix as SAM,
)

# Surface data loading conveniences at the top level as well
from .data import SalamiDataset, SpamDataset, Track


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
    # Label Agreement
    "LAM",
    "SAP",
    "SAM",
    # Data access
    "SalamiDataset",
    "SpamDataset",
    "Track",
    # Submodules (toolboxes)
    "data",
    "ops",
    "viz",
    "core",
    "metrics",
    "exp",
    "lsd",
    "rel",
]
