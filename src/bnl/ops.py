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
    from .core import BoundaryContour, BoundaryHierarchy, MultiSegment

__all__ = [
    "salience_by_counting",
    "default_salience",
    "default_levels",
]


def salience_by_counting(
    ms: MultiSegment,
) -> BoundaryHierarchy:
    """
    Calculates the salience of boundaries based on their frequency of occurrence.

    The salience of each unique boundary time is the number of layers in the
    `MultiSegment` that it appears in.
    """
    # Import here to avoid circular imports
    from .core import BoundaryHierarchy, LeveledBoundary

    # Collect all boundary times and count their frequencies
    time_counts: Counter[float] = Counter()
    for layer in ms.layers:
        for boundary in layer.boundaries:
            time = boundary.time
            time_counts[time] += 1

    # Create rated boundaries with frequency-based salience
    return BoundaryHierarchy(
        boundaries=[LeveledBoundary(time=time, level=count) for time, count in time_counts.items()],
        name=ms.name,
    )


def default_salience(ms: MultiSegment) -> BoundaryHierarchy:
    """
    Calculate the salience of boundaries based on the coarsest layer that they appear in.
    """
    from .core import BoundaryHierarchy, LeveledBoundary

    boundary_map: dict[float, LeveledBoundary] = {}

    # Iterate from finest (last) to coarsest (first) layer.
    # The salience is the layer's rank, starting from 1 for the finest layer.
    # This ensures that if a boundary time exists in multiple layers,
    # the one from the coarsest layer (with highest salience) is kept.
    for salience, layer in enumerate(reversed(ms.layers), start=1):
        for boundary in layer.boundaries:
            boundary_map[boundary.time] = LeveledBoundary(time=boundary.time, level=salience)

    if not boundary_map:
        return BoundaryHierarchy(boundaries=[], name=ms.name)

    return BoundaryHierarchy(boundaries=list(boundary_map.values()), name=ms.name)


def default_levels(bc: BoundaryContour) -> BoundaryHierarchy:
    """
    Find all distinct salience values and use their integer rank as level.
    """
    from .core import BoundaryHierarchy, LeveledBoundary

    # Create a mapping from each unique salience value to its rank (level)
    unique_saliences = sorted({b.salience for b in bc.boundaries[1:-1]})
    max_level = len(unique_saliences)
    sal_level = {sal: lvl for lvl, sal in enumerate(unique_saliences, start=1)}

    # Create LeveledBoundary objects for each boundary in the contour
    inner_boundaries = [LeveledBoundary(time=b.time, level=sal_level[b.salience]) for b in bc.boundaries[1:-1]]

    leveled_boundaries = [
        LeveledBoundary(time=bc.boundaries[0].time, level=max_level),
        *inner_boundaries,
        LeveledBoundary(time=bc.boundaries[-1].time, level=max_level),
    ]

    return BoundaryHierarchy(boundaries=leveled_boundaries, name=bc.name)
