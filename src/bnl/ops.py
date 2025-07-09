"""
This module provides the core algorithmic operations for transforming
boundary and hierarchy objects.

The functions in this module are designed to be composed into pipelines,
either directly or through the fluent API provided by the `bnl.core` classes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from librosa.util import localmax
from mir_eval.hierarchy import _round
from sklearn.neighbors import KernelDensity

from .core import BoundaryContour, BoundaryHierarchy, LeveledBoundary, MultiSegment, RatedBoundary, TimeSpan

# region: Different Notions of Boundary Salience

SalienceStrategy = Literal["count", "depth", "prob"]


def boundary_salience(ms: MultiSegment, strategy: SalienceStrategy = "depth") -> BoundaryContour:
    """
    Calculates boundary salience from a MultiSegment using a specified strategy.

    Parameters
    ----------
    ms : MultiSegment
        The input multi-segment structure.
    strategy : {'depth', 'count', 'prob'}, default 'depth'
        The salience calculation strategy:
        - 'depth': Salience based on the coarsest layer (returns BoundaryHierarchy).
        - 'count': Salience based on frequency (returns BoundaryHierarchy).
        - 'prob': Salience weighted by layer density (returns BoundaryContour).

    Returns
    -------
    BoundaryContour
        The resulting boundary structure. Can be a BoundaryHierarchy if the
        strategy directly produces leveled boundaries.
    """
    if strategy == "depth":
        return _sal_by_depth(ms)
    if strategy == "count":
        return _sal_by_count(ms)
    if strategy == "prob":
        return _sal_by_prob(ms)
    raise ValueError(f"Unknown salience strategy: {strategy}")


def _sal_by_count(ms: MultiSegment) -> BoundaryHierarchy:
    """
    Calculates the salience of boundaries based on their frequency of occurrence.

    The salience of each unique boundary time is the number of layers in the
    `MultiSegment` that it appears in.
    """
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


def _sal_by_depth(ms: MultiSegment) -> BoundaryHierarchy:
    """
    Calculate the salience of boundaries based on the coarsest layer that they appear in.
    """
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


def _sal_by_prob(ms: MultiSegment) -> BoundaryContour:
    """
    Coming from criticism of salience by depth or by count:
    In layers with less boundaries, they (the boundaries) are more important.
    This is because they are intrinsically in a higher level and more salient.
    """
    time_saliences: defaultdict[float, float] = defaultdict(float)

    for layer in ms.layers:
        if len(layer.boundaries) > 2:
            # Weight is inversely proportional to the number of effective boundaries.
            # A layer with fewer boundaries gives more importance to each of its boundaries.
            weight = 1.0 / len(layer.boundaries[1:-1])
            for boundary in layer.boundaries:
                time_saliences[boundary.time] += weight

    rated_boundaries = [RatedBoundary(t, s) for t, s in time_saliences.items()]
    return BoundaryContour(name=ms.name, boundaries=rated_boundaries)


# endregion: Different Notions of Boundary Salience

# region: Two ways to clean up boudnaries closeby in time
BoundaryMergeStrategy = Literal["absorb", "kde"]


def clean_boundaries(
    bc: BoundaryContour, strategy: BoundaryMergeStrategy = "absorb", window: float = 1.0
) -> BoundaryContour:
    """
    Clean up boundaries by removing boundaries that are closeby in time.
    """
    if strategy == "absorb":
        return _boundary_absorb(bc, window)
    if strategy == "kde":
        return _boundary_peakpick_kde(bc, window)
    raise ValueError(f"Unknown boundary merge strategy: {strategy}")


def _boundary_absorb(bc: BoundaryContour, window: float = 1.0) -> BoundaryContour:
    """
    Merge boundaries by absorbing them into boundaries that's more salient.
    Boundaries tied by salience are first come first serve in time of appearance.
    """
    if len(bc.boundaries) == 2:
        return bc

    outer_boundaries = [bc.boundaries[0], bc.boundaries[-1]]
    inner_boundaries = sorted(bc.boundaries[1:-1], key=lambda b: b.salience, reverse=True)

    kept_boundaries = list(outer_boundaries)
    for new_b in inner_boundaries:
        # A less salient boundary is absorbed if it's too close to any more salient one.
        is_absorbed = any(abs(new_b.time - kept_b.time) <= window for kept_b in kept_boundaries)
        if not is_absorbed:
            kept_boundaries.append(new_b)

    boundaries = [RatedBoundary(b.time, b.salience) for b in sorted(kept_boundaries)]
    return BoundaryContour(name=bc.name, boundaries=boundaries)


def _boundary_peakpick_kde(bc: BoundaryContour, window: float = 1.0, frame_size: float = 0.1) -> BoundaryContour:
    """
    Merge boundaries by finding peaks in a weighted kernel density estimate.
    """
    # KDE is not meaningful for fewer than 2 inner boundaries.
    if len(bc.boundaries) < 4:
        return bc

    inner_boundaries = bc.boundaries[1:-1]
    times = np.array([b.time for b in inner_boundaries]).reshape(-1, 1)
    saliences = np.array([b.salience for b in inner_boundaries])

    # Use scikit-learn's KernelDensity which supports sample weights.
    kde = KernelDensity(kernel="gaussian", bandwidth=window)
    kde.fit(times, sample_weight=saliences)

    # Evaluate the KDE on a fine grid to find peaks.
    grid_times = _build_time_grid(bc, frame_size=frame_size)
    log_density = kde.score_samples(grid_times)
    density = np.exp(log_density)

    # Find peaks in the density curve using librosa's local maxima detection.
    peak_indices = localmax(density)
    peak_times = grid_times.flatten()[peak_indices]
    peak_saliences = density[peak_indices]
    max_salience = np.max(peak_saliences)

    # Create new boundaries from the found peaks.
    new_inner_boundaries = [RatedBoundary(time=t, salience=s) for t, s in zip(peak_times, peak_saliences, strict=True)]

    # Combine with original outer boundaries and return a new contour.
    final_boundaries = [
        RatedBoundary(bc.start.time, salience=max_salience),
        *new_inner_boundaries,
        RatedBoundary(bc.end.time, salience=max_salience),
    ]

    return BoundaryContour(name=bc.name, boundaries=final_boundaries)


def _build_time_grid(span: TimeSpan, frame_size: float = 0.1) -> np.ndarray:
    """
    Build a grid of times using the same logic as mir_eval to build the ticks
    """
    # Figure out how many frames we need
    n_frames = int((_round(span.end.time, frame_size) - _round(span.start.time, frame_size)) / frame_size)
    return np.arange(n_frames + 1) * frame_size + span.start.time


# endregion: Two ways to clean up boudnaries closeby in time

# region: Stratification into levels


def level_by_distinct_salience(bc: BoundaryContour) -> BoundaryHierarchy:
    """
    Find all distinct salience values and use their integer rank as level.
    """
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


# endregion: Stratification into levels
