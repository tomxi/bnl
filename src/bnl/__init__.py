"""BNL: A Python library for hierarchical text segmentation and evaluation.

This library provides tools for working with hierarchical text segments.

Submodules
----------
core
    Core data structures and functionality.
viz
    Visualization utilities for segmentations.
data
    Data loading and management for musical structure datasets.
ops
    Operations and transformations on BNL objects.
metrics
    Evaluation metrics for segmentations.
"""

__version__ = "0.2.1"

# --- Import submodules for explicit, namespaced access ---
from . import data, metrics, ops, viz

# --- Promote the core data structures to the top level ---
from .core import Hierarchy, Segmentation, TimeSpan

# --- Define the public-facing API of the `bnl` package ---
__all__ = [
    # Promoted from .core
    "TimeSpan",
    "Segmentation",
    "Hierarchy",
    # Submodules (toolboxes)
    "viz",
    "data",
    "ops",
    "metrics",
]
