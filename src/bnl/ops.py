"""
This module provides the core algorithmic operations for transforming
boundary and hierarchy objects.

The functions in this module are designed to be composed into pipelines,
either directly or through the fluent API provided by the `bnl.core` classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from librosa.util import localmax
from mir_eval.hierarchy import _round
from sklearn.neighbors import KernelDensity

from .core import BoundaryContour, BoundaryHierarchy, LeveledBoundary, MultiSegment, RatedBoundary, TimeSpan

# region: Different Notions of Boundary Salience


def boundary_salience(ms: MultiSegment, strategy: str = "depth") -> BoundaryContour:
    """
    runs boundary salience from a MultiSegment using a specified strategy.

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
    if strategy not in _SalienceStrategy._registry:
        raise ValueError(f"Unknown salience strategy: {strategy}")
    return _SalienceStrategy._registry[strategy](ms)


class _SalienceStrategy(ABC):
    """Abstract base class for salience calculation strategies."""

    _registry: dict[str, _SalienceStrategy] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[_SalienceStrategy]], type[_SalienceStrategy]]:
        """Registers a strategy in a central registry."""

        def decorator(strategy_cls: type[_SalienceStrategy]) -> type[_SalienceStrategy]:
            cls._registry[name] = strategy_cls()
            return strategy_cls

        return decorator

    @abstractmethod
    def __call__(self, ms: MultiSegment) -> BoundaryContour:
        raise NotImplementedError


@_SalienceStrategy.register("count")
class _SalByCount(_SalienceStrategy):
    """Salience based on frequency of occurrence."""

    def __call__(self, ms: MultiSegment) -> BoundaryHierarchy:
        """
        The salience of each unique boundary time is the number of layers in the
        `MultiSegment` that it appears in.
        """
        time_counts: Counter[float] = Counter(b.time for layer in ms.layers for b in layer.boundaries)
        return BoundaryHierarchy(
            boundaries=[LeveledBoundary(time=time, level=count) for time, count in time_counts.items()],
            name=ms.name,
        )


@_SalienceStrategy.register("depth")
class _SalByDepth(_SalienceStrategy):
    """Salience based on the coarsest layer of appearance."""

    def __call__(self, ms: MultiSegment) -> BoundaryHierarchy:
        """
        Iterate from finest (last) to coarsest (first) layer.
        The salience is the layer's rank, starting from 1 for the finest layer.
        This ensures that if a boundary time exists in multiple layers,
        the one from the coarsest layer (with highest salience) is kept.
        """
        boundary_map: dict[float, LeveledBoundary] = {}
        for salience, layer in enumerate(reversed(ms.layers), start=1):
            for boundary in layer.boundaries:
                boundary_map[boundary.time] = LeveledBoundary(time=boundary.time, level=salience)
        return BoundaryHierarchy(boundaries=list(boundary_map.values()), name=ms.name)


@_SalienceStrategy.register("prob")
class _SalByProb(_SalienceStrategy):
    """Salience weighted by layer density."""

    def __call__(self, ms: MultiSegment) -> BoundaryContour:
        """
        In layers with less boundaries, they are more important.
        This is because they are intrinsically in a higher level and more salient.
        """
        time_saliences: defaultdict[float, float] = defaultdict(float)
        for layer in ms.layers:
            if len(layer.boundaries) > 2:
                # Weight is inversely proportional to the number of effective boundaries.
                weight = 1.0 / len(layer.boundaries[1:-1])
                for boundary in layer.boundaries:
                    time_saliences[boundary.time] += weight
        return BoundaryContour(
            name=ms.name,
            boundaries=[RatedBoundary(t, s) for t, s in time_saliences.items()],
        )


# endregion: Different Notions of Boundary Salience

# region: Two ways to clean up boundaries closeby in time


class _CleanStrategy(ABC):
    """Abstract base class for boundary cleaning strategies."""

    _registry: dict[str, type[_CleanStrategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[_CleanStrategy]], type[_CleanStrategy]]:
        """A class method to register strategies in a central registry."""

        def decorator(strategy_cls: type[_CleanStrategy]) -> type[_CleanStrategy]:
            cls._registry[name] = strategy_cls
            return strategy_cls

        return decorator

    @abstractmethod
    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        """Cleans boundaries in a BoundaryContour."""
        raise NotImplementedError


@_CleanStrategy.register("absorb")
class _CleanByAbsorb(_CleanStrategy):
    """Clean boundaries by absorbing less salient ones within a window."""

    def __init__(self, window: float = 1.0) -> None:
        self.window = window

    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        from .core import BoundaryContour, RatedBoundary

        if len(bc.boundaries) <= 2:
            return bc

        outer_boundaries = [bc.boundaries[0], bc.boundaries[-1]]
        inner_boundaries = sorted(bc.boundaries[1:-1], key=lambda b: b.salience, reverse=True)

        kept_boundaries = list(outer_boundaries)
        for new_b in inner_boundaries:
            is_absorbed = any(abs(new_b.time - kept_b.time) <= self.window for kept_b in kept_boundaries)
            if not is_absorbed:
                kept_boundaries.append(new_b)

        boundaries = [RatedBoundary(b.time, b.salience) for b in sorted(kept_boundaries)]
        return BoundaryContour(name=bc.name, boundaries=boundaries)


@_CleanStrategy.register("kde")
class _CleanByKDE(_CleanStrategy):
    """Clean boundaries by finding peaks in a weighted kernel density estimate."""

    def __init__(self, bandwidth: float = 1.0, frame_size: float = 0.1):
        self.kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.frame_size = frame_size

    def _build_time_grid(self, span: TimeSpan) -> np.ndarray:
        """
        Build a grid of times using the same logic as mir_eval to build the ticks
        """
        # Figure out how many frames we need by using `mir_eval`'s exact frame finding logic.
        n_frames = int(
            (_round(span.end.time, self.frame_size) - _round(span.start.time, self.frame_size)) / self.frame_size
        )
        return np.arange(n_frames + 1) * self.frame_size + span.start.time

    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        if len(bc.boundaries) < 4:
            return bc

        inner_boundaries = bc.boundaries[1:-1]
        times = np.array([b.time for b in inner_boundaries]).reshape(-1, 1)
        saliences = np.array([b.salience for b in inner_boundaries])

        self.kde.fit(times, sample_weight=saliences)

        grid_times = self._build_time_grid(bc)
        density = np.exp(self.kde.score_samples(grid_times.reshape(-1, 1)))

        peak_indices = localmax(density)
        peak_times = grid_times.flatten()[peak_indices]
        peak_saliences = np.exp(self.kde.score_samples(peak_times.reshape(-1, 1)))
        max_salience = np.max(peak_saliences) if peak_saliences.size > 0 else 0

        new_inner_boundaries = [RatedBoundary(t, s) for t, s in zip(peak_times, peak_saliences, strict=True)]
        final_boundaries = [
            RatedBoundary(bc.start.time, max_salience),
            *new_inner_boundaries,
            RatedBoundary(bc.end.time, max_salience),
        ]
        return BoundaryContour(name=bc.name, boundaries=final_boundaries)


def clean_boundaries(bc: BoundaryContour, strategy: str = "absorb", **kwargs: Any) -> BoundaryContour:
    """
    Clean up boundaries by removing boundaries that are closeby in time.
    """
    if strategy not in _CleanStrategy._registry:
        raise ValueError(f"Unknown boundary cleaning strategy: {strategy}")

    strategy_instance = _CleanStrategy._registry[strategy](**kwargs)
    return strategy_instance(bc)


# endregion: Two ways to clean up boundaries closeby in time

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
