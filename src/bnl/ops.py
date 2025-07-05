"""
This module provides the core algorithmic operations for transforming
boundary and hierarchy objects.

The functions in this module are designed to be composed into pipelines,
either directly or through the fluent API provided by the `bnl.core` classes.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import MultiSegment


def calculate_frequency_salience(
    mseg: MultiSegment,
) -> dict[float, int]:
    """
    Calculates the salience of boundaries based on their frequency of occurrence.

    The salience of each unique boundary time is the number of layers in the
    `MultiSegment` that it appears in.

    Args:
        mseg: The `MultiSegment` to analyze.

    Returns:
        A dictionary mapping each boundary time to its frequency count.
    """
    all_boundaries = [b.time for layer in mseg.layers for b in layer.boundaries]
    return Counter(all_boundaries)
