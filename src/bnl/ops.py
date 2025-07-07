"""
This module provides the core algorithmic operations for transforming
boundary and hierarchy objects.

The functions in this module are designed to be composed into pipelines,
either directly or through the fluent API provided by the `bnl.core` classes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import BoundaryContour, BoundaryHierarchy, MultiSegment

__all__ = [
    "salience_by_counting",
    "default_salience",
    "default_levels",
    "default_labeling",
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

    if not bc.boundaries:
        return BoundaryHierarchy(boundaries=[], name=bc.name)

    # Create a mapping from each unique salience value to its rank (level)
    unique_saliences = sorted({b.salience for b in bc.boundaries})
    salience_to_level = {sal: lvl + 1 for lvl, sal in enumerate(unique_saliences)}

    # Create LeveledBoundary objects for each boundary in the contour
    leveled_boundaries = [LeveledBoundary(time=b.time, level=salience_to_level[b.salience]) for b in bc.boundaries]

    return BoundaryHierarchy(boundaries=leveled_boundaries, name=bc.name)


def default_labeling(bh: BoundaryHierarchy) -> MultiSegment:
    """
    Simply label each timespan with its default string representation.
    """

    from .core import Boundary, MultiSegment, Segment

    # For each level, collect all boundaries that exist at or above that level.
    # A boundary at level N exists in all layers from N down to 1.
    boundaries_by_level = defaultdict(set)
    for b in bh.boundaries:
        for level in range(1, b.level + 1):
            boundaries_by_level[level].add(Boundary(time=b.time))

    # Build layers from coarsest (highest level) to finest (lowest level).
    unique_levels = sorted(boundaries_by_level.keys(), reverse=True)

    layers = []
    for level in unique_levels:
        # Sort the boundaries for the current level to form a valid segment.
        level_boundaries = sorted(list(boundaries_by_level[level]), key=lambda b: b.time)
        labels = [""] * len(level_boundaries[:-1])
        layers.append(Segment(boundaries=level_boundaries, labels=labels, name=f"Level {level}"))

    return MultiSegment(layers=layers, name=bh.name)
