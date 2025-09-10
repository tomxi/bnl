"""
This module provides the core algorithmic operations for transforming

The core components are:

- A generic `Strategy` base class that provides a registry pattern.
- Three specialized abstract strategy classes that inherit from `Strategy`:

  - `SalienceStrategy`: For calculating boundary importance.
  - `CleanStrategy`: For refining boundary contours.
  - `LevelStrategy`: For converting continuous salience into discrete levels.

- Concrete implementations for each strategy type, which can be extended.
"""

from __future__ import annotations

__all__ = [
    "SalienceStrategy",
    "SalByCount",
    "SalByDepth",
    "SalByProb",
    "CleanStrategy",
    "CleanByAbsorb",
    "CleanByKDE",
    "LevelStrategy",
    "LevelByUniqueSal",
    "LevelByMeanShift",
    "LabelAgreementStrategy",
    "LamByDepth",
    "LamByProb",
    "LamByCount",
]

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import frameless_eval as fle
import mir_eval
import numpy as np
import scipy.signal
from mir_eval.hierarchy import _round
from sklearn.cluster import MeanShift
from sklearn.neighbors import KernelDensity

from .core import (
    BoundaryContour,
    BoundaryHierarchy,
    LeveledBoundary,
    MultiSegment,
    RatedBoundary,
    TimeSpan,
)

# region: Base Strategy Pattern


class Strategy(ABC):
    """Abstract base class for all strategy patterns in BNL."""

    # This is a placeholder; each subclass should have its own registry.
    _registry: dict[str, type[Strategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Strategy]], type[Strategy]]:
        """A class method to register strategies in a central registry."""

        def decorator(strategy_cls: type[Strategy]) -> type[Strategy]:
            cls._registry[name] = strategy_cls
            return strategy_cls

        return decorator

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError  # pragma: no cover


# endregion: Base Strategy Pattern


# region: Different Notions of Boundary Salience
class SalienceStrategy(Strategy):
    """Abstract base class for salience calculation strategies."""

    _registry: dict[str, type[SalienceStrategy]] = {}

    @abstractmethod
    def __call__(self, ms: MultiSegment) -> BoundaryContour:
        raise NotImplementedError  # pragma: no cover


@SalienceStrategy.register("count")
class SalByCount(SalienceStrategy):
    """Salience based on frequency of occurrence."""

    def __call__(self, ms: MultiSegment) -> BoundaryHierarchy:
        """
        The salience of each unique boundary time is the number of layers in the
        `MultiSegment` that it appears in.
        """
        time_counts: Counter[float] = Counter(b.time for layer in ms.layers for b in layer.bs)
        return BoundaryHierarchy(
            bs=sorted(
                [LeveledBoundary(time=time, level=count) for time, count in time_counts.items()]
            ),
            name=ms.name or "Salience Hierarchy",
        )


@SalienceStrategy.register("depth")
class SalByDepth(SalienceStrategy):
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
            for boundary in layer.bs:
                boundary_map[boundary.time] = LeveledBoundary(time=boundary.time, level=salience)
        return BoundaryHierarchy(
            bs=sorted(list(boundary_map.values())), name=ms.name or "Salience Hierarchy"
        )


@SalienceStrategy.register("prob")
class SalByProb(SalienceStrategy):
    """Salience weighted by layer density."""

    def __call__(self, ms: MultiSegment) -> BoundaryContour:
        """
        In layers with less boundaries, they are more important.
        This is because they are intrinsically in a higher level and more salient.
        """
        time_saliences: defaultdict[float, float] = defaultdict(float)
        for layer in ms.layers:
            if len(layer.bs) > 2:
                # Weight is inversely proportional to the number of effective boundaries.
                weight = 1.0 / len(layer.bs[1:-1])
                for boundary in layer.bs:
                    time_saliences[boundary.time] += weight
        return BoundaryContour(
            name=ms.name or "Salience Contour",
            bs=sorted([RatedBoundary(t, s) for t, s in time_saliences.items()]),
        )


# endregion: Different Notions of Boundary Salience


# region: Two ways to clean up boundaries closeby in time


class CleanStrategy(Strategy):
    """Abstract base class for boundary cleaning strategies."""

    _registry: dict[str, type[CleanStrategy]] = {}

    @abstractmethod
    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        """Cleans boundaries in a BoundaryContour."""
        raise NotImplementedError  # pragma: no cover


@CleanStrategy.register("absorb")
class CleanByAbsorb(CleanStrategy):
    """Clean boundaries by absorbing less salient ones within a window."""

    def __init__(self, window: float = 1.5) -> None:
        self.window = window

    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        if len(bc.bs) <= 2:
            return bc

        inner_boundaries = sorted(bc.bs[1:-1], key=lambda b: b.salience, reverse=True)

        kept_boundaries = [bc.start, bc.end]
        for new_b in inner_boundaries:
            is_absorbed = any(
                abs(new_b.time - kept_b.time) <= self.window for kept_b in kept_boundaries
            )
            if not is_absorbed:
                kept_boundaries.append(new_b)

        boundaries = [RatedBoundary(b.time, b.salience) for b in sorted(kept_boundaries)]
        return BoundaryContour(name=bc.name or "Cleaned Contour", bs=boundaries)


@CleanStrategy.register("kde")
class CleanByKDE(CleanStrategy):
    """Clean boundaries by finding peaks in a weighted kernel density estimate."""

    def __init__(self, bw: float = 0.5):
        self.time_kde = KernelDensity(kernel="gaussian", bandwidth=bw)

    def _build_time_grid(self, span: TimeSpan, frame_size: float = 0.1) -> np.ndarray:
        """
        Build a grid of times using the same logic as mir_eval to build the ticks
        """
        # Figure out how many frames we need by using `mir_eval`'s exact frame finding logic.
        n_frames = int(
            (_round(span.end.time, frame_size) - _round(span.start.time, frame_size)) / frame_size
        )
        return np.arange(n_frames + 1) * frame_size + span.start.time

    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        if len(bc.bs) < 4:  # if only 3 boundaries (1 start, 1 end, 1 inner), just return
            return bc

        inner_boundaries = bc.bs[1:-1]
        times = np.array([b.time for b in inner_boundaries])
        saliences = np.array([b.salience for b in inner_boundaries])

        self.time_kde.fit(times.reshape(-1, 1), sample_weight=saliences)

        grid_times = self._build_time_grid(bc, frame_size=0.1)
        log_density = self.time_kde.score_samples(grid_times.reshape(-1, 1))

        peak_indices = scipy.signal.find_peaks(log_density)[0]
        peak_times = grid_times.flatten()[peak_indices]
        peak_saliences = np.exp(log_density[peak_indices])
        max_salience = np.max(peak_saliences) if peak_saliences.size > 0 else 1

        new_inner_boundaries = [
            RatedBoundary(t, s) for t, s in zip(peak_times, peak_saliences, strict=True)
        ]
        final_boundaries = [
            RatedBoundary(bc.start.time, max_salience),
            *new_inner_boundaries,
            RatedBoundary(bc.end.time, max_salience),
        ]
        return BoundaryContour(name=bc.name or "Cleaned Contour", bs=sorted(final_boundaries))


@CleanStrategy.register("none")
class CleanNone(CleanStrategy):
    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        return bc


# endregion: Two ways to clean up boundaries closeby in time


# region: Stratification into levels
class LevelStrategy(Strategy):
    """Abstract base class for boundary level quantizing strategies."""

    _registry: dict[str, type[LevelStrategy]] = {}

    @abstractmethod
    def __call__(self, bc: BoundaryContour) -> BoundaryHierarchy:
        """Quantize boundaries' levels in a BoundaryContour."""
        raise NotImplementedError  # pragma: no cover


@LevelStrategy.register("unique")
class LevelByUniqueSal(LevelStrategy):
    """
    Find all distinct salience values and use their integer rank as level.
    """

    def __call__(self, bc: BoundaryContour) -> BoundaryHierarchy:
        # Create a mapping from each unique salience value to its rank (level)
        unique_saliences = sorted({b.salience for b in bc.bs[1:-1]})
        max_level = len(unique_saliences) if unique_saliences else 1
        sal_level = {sal: lvl for lvl, sal in enumerate(unique_saliences, start=1)}

        # Create LeveledBoundary objects for each boundary in the contour
        inner_boundaries = [
            LeveledBoundary(time=b.time, level=sal_level[b.salience]) for b in bc.bs[1:-1]
        ]

        return BoundaryHierarchy(
            bs=[
                LeveledBoundary(time=bc.bs[0].time, level=max_level),
                *inner_boundaries,
                LeveledBoundary(time=bc.bs[-1].time, level=max_level),
            ],
            name=bc.name or "Unique Salience Hierarchy",
        )


@LevelStrategy.register("mean_shift")
class LevelByMeanShift(LevelStrategy):
    """
    Use mean shift clustering to find peaks in the salience values and clusters them into levels.
    """

    def __init__(self, bw: float = 0.15):
        self.sal_ms = MeanShift(bandwidth=bw)

    def __call__(self, bc: BoundaryContour) -> BoundaryHierarchy:
        if len(bc.bs) < 4:  # if only 3 boundaries (1 start, 1 end, 1 inner), just return
            return bc.level(strategy="unique")

        inner_boundaries = bc.bs[1:-1]
        saliences = np.array([b.salience for b in inner_boundaries])
        normalized_saliences = saliences / saliences.max()
        self.sal_ms.fit(normalized_saliences.reshape(-1, 1))
        quantized_salience = self.sal_ms.cluster_centers_.flatten()[self.sal_ms.labels_]
        inner_boundaries = [
            RatedBoundary(time=b.time, salience=s)
            for b, s in zip(inner_boundaries, quantized_salience)
        ]
        quantized_bc = BoundaryContour(bs=[bc.start, *inner_boundaries, bc.end], name=bc.name)
        return quantized_bc.level(strategy="unique")


# endregion: Stratification into levels


# region: Label Agreement Strategies
class LabelAgreementStrategy(Strategy):
    """Abstract base class for label agreement strategies."""

    _registry: dict[str, type[LabelAgreementStrategy]] = {}

    @abstractmethod
    def __call__(self, ms: MultiSegment) -> tuple[np.ndarray, np.ndarray]:
        """Combining label agreement maps of a hierarchy.
        returns (boundaries (1D array), label_agreement_map (2D array))
        """
        raise NotImplementedError  # pragma: no cover


@LabelAgreementStrategy.register("depth")
class LamByDepth(LabelAgreementStrategy):
    """Combining label agreement maps of a hierarchy using depth.

    mono: if True, force monotonicity by making sure it is showing
          the shallowest level where labels stop meeting.
          default: False is the conventional meet matrix: deepest level where labels meet.
    """

    def __init__(self, mono: bool = False):
        self.mono = mono

    def __call__(self, ms: MultiSegment) -> tuple[np.ndarray, np.ndarray]:
        grid, labels, _ = fle.utils.make_common_itvls(ms.itvls, ms.labels, [], [])
        boundaries = mir_eval.util.intervals_to_boundaries(grid)
        return boundaries, fle.utils.meet(labels, mono=self.mono)


@LabelAgreementStrategy.register("count")
class LamByCount(LabelAgreementStrategy):
    """Combining label agreement maps of a hierarchy using count."""

    def __call__(self, ms: MultiSegment) -> tuple[np.ndarray, np.ndarray]:
        grid, labels, _ = fle.utils.make_common_itvls(ms.itvls, ms.labels, [], [])
        labels = np.asarray(labels)
        # Using broadcasting to compute the outer comparison for each level.
        meet_per_level = np.equal(labels[:, :, None], labels[:, None, :])
        meet_lvl_count = np.sum(meet_per_level, axis=0)
        boundaries = mir_eval.util.intervals_to_boundaries(grid)
        return boundaries, meet_lvl_count


@LabelAgreementStrategy.register("prob")
class LamByProb(LabelAgreementStrategy):
    """Combining label agreement maps of a hierarchy using lam prob density."""

    def __call__(self, ms: MultiSegment) -> tuple[np.ndarray, np.ndarray]:
        grid, labels, _ = fle.utils.make_common_itvls(ms.itvls, ms.labels, [], [])
        labels = np.asarray(labels)
        sec_dur = grid[:, 1] - grid[:, 0]
        grid_area = np.outer(sec_dur, sec_dur)
        # Using broadcasting to compute the outer comparison for each level.
        # First dim is depth, 2nd and 3rd dim are segment indices
        meet_per_level = np.equal(labels[:, :, None], labels[:, None, :])

        # add a dim to grid_area to scale meet mats by segment duration ^ 2
        meet_area_per_level = meet_per_level * grid_area[None, :, :]
        total_meet_area_per_level = np.sum(meet_area_per_level, axis=(1, 2))
        # numbers will get very small, so use portions as opposed to raw meet area in seconds^2
        # This way the numbers will be comparable for tracks of different lengths
        meet_portion_per_level = total_meet_area_per_level / np.sum(grid_area)
        level_lam_pdfs = meet_per_level / meet_portion_per_level[:, None, None]

        # Boundaries too
        boundaries = mir_eval.util.intervals_to_boundaries(grid)
        return boundaries, np.mean(level_lam_pdfs, axis=0)


# endregion: Label Agreement Strategies
