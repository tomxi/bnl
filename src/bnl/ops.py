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
    # boundary salience a.k.a boundary prominence
    "SalienceStrategy",
    "SalByCount",
    "SalByDepth",
    "SalByProb",
    # boundary cleaning
    "CleanStrategy",
    "CleanByAbsorb",
    "CleanByKDE",
    # boundary level assignment / quantization
    "LevelStrategy",
    "LevelByUniqueSal",
    "LevelByMeanShift",
    # label agreement map building
    "LabelAgreementStrategy",
    "LamByDepth",
    "LamByProb",
    "LamByCount",
    # Labeling strategies
    "LabelingStrategy",
    "LabelByUniqueLabel",
    "LabelByClosestSingleLayer",
    "LabelByLam",
]

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import frameless_eval as fle
import mir_eval
import numpy as np
import pandas as pd
import scipy.signal
from mir_eval.hierarchy import _round
from sklearn.cluster import MeanShift

from .core import (
    Boundary,
    BoundaryContour,
    BoundaryHierarchy,
    LabelAgreementMap,
    LeveledBoundary,
    MultiSegment,
    RatedBoundary,
    Segment,
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

    def __init__(self, w: pd.Series | None = None) -> None:
        self.layer_w = w

    def __call__(self, ms: MultiSegment) -> BoundaryContour:
        """
        In layers with less boundaries, they are more important.
        This is because they are intrinsically in a higher level and more salient.
        """
        if self.layer_w is None:
            uniform_weights = np.ones(len(ms.layers)) / len(ms.layers)
            self.layer_w = pd.Series(index=ms.layer_names, data=uniform_weights)
        time_saliences: defaultdict[float, float] = defaultdict(float)
        for layer in ms.layers:
            if len(layer.bs) > 2 and self.layer_w[layer.name] > 0:
                # Weight is inversely proportional to the number of effective boundaries.
                weight = 1.0 / len(layer.bs[1:-1])
                for boundary in layer.bs:
                    time_saliences[boundary.time] += weight * self.layer_w[layer.name]
        # manually add in the first and last boundaries, matching the highest probability
        if len(time_saliences) == 0:
            max_prob = 1.0
        else:
            max_prob = max(time_saliences.values())
        time_saliences[ms.start.time] = max_prob
        time_saliences[ms.end.time] = max_prob

        return BoundaryContour(
            name=ms.name or "Salience Contour",
            bs=sorted([RatedBoundary(t, s) for t, s in time_saliences.items()]),
        )


# endregion: Different Notions of Boundary Salience


# region: Clean up boundaries closeby in time


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

    def __init__(self, window: float = 1.0) -> None:
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

    def __init__(self, bw: float = 0.5, frame_size: float = 0.1) -> None:
        self.bw = bw
        self.frame_size = frame_size

    def __call__(self, bc: BoundaryContour) -> BoundaryContour:
        if len(bc.bs) <= 3:  # if only 3 boundaries (1 start, 1 end, 1 inner), just return
            return bc

        grid_times = build_time_grid(bc, frame_size=self.frame_size)
        bpc = bc.bpc(bw=self.bw, time_grid=grid_times)
        boundaries = bpc2bs(bpc, bc.start.time, bc.end.time)
        return BoundaryContour(name=bc.name or "Cleaned Contour", bs=boundaries)


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

    def __init__(self, bw: float = 1 / 3, log: bool = True):
        self.log = log
        if log:
            bw = -np.log(1 - bw)
            # print(bw)
        self.sal_ms = MeanShift(bandwidth=bw)

    def __call__(self, bc: BoundaryContour) -> BoundaryHierarchy:
        if len(bc.bs) < 4:  # if only 3 boundaries (1 start, 1 end, 1 inner), just return
            return bc.level(strategy="unique")

        inner_boundaries = bc.bs[1:-1]
        saliences = np.array([b.salience for b in inner_boundaries])
        if self.log:
            saliences = np.log(saliences)
        else:
            saliences /= saliences.max()

        self.sal_ms.fit(saliences.reshape(-1, 1))
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
    def __call__(self, ms: MultiSegment) -> LabelAgreementMap:
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

    def __call__(self, ms: MultiSegment) -> LabelAgreementMap:
        grid, labels, _ = fle.utils.make_common_itvls(ms.itvls, ms.labels, [], [])
        boundaries = mir_eval.util.intervals_to_boundaries(grid)
        return LabelAgreementMap(bs=boundaries, mat=fle.utils.meet(labels, mono=self.mono))


@LabelAgreementStrategy.register("count")
class LamByCount(LabelAgreementStrategy):
    """Combining label agreement maps of a hierarchy using count."""

    def __call__(self, ms: MultiSegment) -> LabelAgreementMap:
        grid, labels, _ = fle.utils.make_common_itvls(ms.itvls, ms.labels, [], [])
        labels = np.asarray(labels)
        # Using broadcasting to compute the outer comparison for each level.
        meet_per_level = np.equal(labels[:, :, None], labels[:, None, :])
        meet_lvl_count = np.sum(meet_per_level, axis=0)
        boundaries = mir_eval.util.intervals_to_boundaries(grid)
        return LabelAgreementMap(bs=boundaries, mat=meet_lvl_count)


@LabelAgreementStrategy.register("prob")
class LamByProb(LabelAgreementStrategy):
    """Combining label agreement maps of a hierarchy using lam prob density."""

    def __init__(self, w: pd.Series | None = None):
        # We are multiplying by len(w) because we are doing np.mean later
        self.w = w / (w.sum() + 1e-10) * len(w) if w is not None else None

    def __call__(self, ms: MultiSegment) -> LabelAgreementMap:
        grid, labels, _ = fle.utils.make_common_itvls(ms.itvls, ms.labels, [], [])
        labels = np.asarray(labels)
        seg_dur = grid[:, 1] - grid[:, 0]
        grid_area = np.outer(seg_dur, seg_dur)
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

        # weighted combination
        if self.w is not None:
            level_lam_pdfs = level_lam_pdfs * self.w.values[:, None, None]
        return LabelAgreementMap(bs=boundaries, mat=np.mean(level_lam_pdfs, axis=0))


# endregion: Label Agreement Strategies

# region: Label filling strategies


class LabelingStrategy(Strategy):
    """Abstract base class for label filling strategies."""

    _registry: dict[str, type[LabelingStrategy]] = {}

    @abstractmethod
    def __call__(self, bh: BoundaryHierarchy) -> MultiSegment:
        raise NotImplementedError  # pragma: no cover


@LabelingStrategy.register("unique")
class LabelByUniqueLabel(LabelingStrategy):
    """Fill empty strings with the closest single layer label."""

    def __call__(self, bh: BoundaryHierarchy) -> MultiSegment:
        layers = []
        max_level = max(b.level for b in bh.bs)
        for level in range(max_level, 0, -1):
            level_boundaries = [Boundary(b.time) for b in bh.bs if b.level >= level]
            labels = [None] * (len(level_boundaries) - 1)
            layers.append(
                Segment(
                    bs=level_boundaries,
                    raw_labs=labels,
                    name=f"L{max_level - level + 1:02d}",
                )
            )

        return MultiSegment(raw_layers=layers, name=bh.name)


@LabelingStrategy.register("1layer")
class LabelByClosestSingleLayer(LabelingStrategy):
    """Find the closest single layer from a given MultiSegment, for each BH level
    and fill the labels by looking for largest overlap."""

    def __init__(self, ref_ms: MultiSegment, metric="v", hr_window=1):
        self.reference_ms = ref_ms
        self.metric = metric  # "hr" or "v"
        self.hr_window = hr_window

    def __call__(self, bh: BoundaryHierarchy) -> MultiSegment:
        # get the MS with empty strings first
        empty_ms = LabelByUniqueLabel()(bh)
        labels = []
        ref_layers_used = []
        for layer in empty_ms.layers:
            # Find the ref_ms layer that has highest boundary HR score with current level_boundaries
            if self.metric == "hr":
                best_ref_layer = max(
                    self.reference_ms.layers,
                    key=lambda ref_layer: mir_eval.segment.detection(
                        ref_layer.itvls, layer.itvls, window=self.hr_window
                    )[2],
                )
            elif self.metric == "v":
                best_ref_layer = max(
                    self.reference_ms.layers,
                    key=lambda ref_layer: fle.vmeasure(
                        ref_layer.itvls,
                        np.arange(len(ref_layer)).astype(str).tolist(),
                        layer.itvls,
                        np.arange(len(layer)).astype(str).tolist(),
                    )[2],
                )
            ref_layers_used.append(best_ref_layer)
            # find the max overlapping reference interval and use their labels
            layer_labels = self.find_max_overlap_labels(
                best_ref_layer.btimes,
                best_ref_layer.labels,
                layer.btimes,
            )
            labels.append(layer_labels)

        return MultiSegment.from_itvls(empty_ms.itvls, labels, name=bh.name)

    @staticmethod
    def find_max_overlap_labels(ref_boundaries, ref_labels, est_boundaries) -> np.ndarray:
        """
        For each estimated interval, find the reference interval index with max overlap.

        Args:
            ref_boundaries: A sorted np array of boundary times for reference segments.
            est_boundaries: A sorted np array of boundary times for estimated segments.

        Returns:
            An array of indices, where the i-th element is the index of the reference
            interval that maximally overlaps with the i-th estimated interval.
        """
        # Convert boundaries to interval start and end times
        ref_starts, ref_ends = np.array(ref_boundaries[:-1]), np.array(ref_boundaries[1:])
        est_starts, est_ends = np.array(est_boundaries[:-1]), np.array(est_boundaries[1:])

        # For each estimated interval, find the range of candidate reference intervals
        # `start_indices`: First ref interval that could possibly overlap
        # `end_indices`: First ref interval that is guaranteed to be past the est interval
        start_indices = np.searchsorted(ref_ends, est_starts, side="right")
        end_indices = np.searchsorted(ref_starts, est_ends, side="left")

        max_overlap_labels = []
        # Iterate through each estimated interval and its candidate reference intervals
        for i, (est_s, est_e) in enumerate(zip(est_starts, est_ends)):
            # Slice the candidate reference intervals for the current estimated interval
            cand_start, cand_end = start_indices[i], end_indices[i]
            cand_ref_starts = ref_starts[cand_start:cand_end]
            cand_ref_ends = ref_ends[cand_start:cand_end]
            cand_ref_labels = ref_labels[cand_start:cand_end]

            # Calculate overlap for all candidates in a vectorized way
            # overlap = max(0, min(end1, end2) - max(start1, start2))
            overlaps = np.maximum(
                0, np.minimum(est_e, cand_ref_ends) - np.maximum(est_s, cand_ref_starts)
            )

            # Sum overlaps by label using a defaultdict
            label_overlaps = defaultdict(float)
            for label, overlap in zip(cand_ref_labels, overlaps):
                if overlap > 0:
                    label_overlaps[label] += overlap

            # Find the label with the maximum summed overlap
            best_label = max(label_overlaps, key=label_overlaps.get)
            max_overlap_labels.append(best_label)

        return max_overlap_labels


@LabelingStrategy.register("lam")
class LabelByLam(LabelingStrategy):
    """Use LAM and eigengap spectral clustering to label intervals."""

    def __init__(
        self,
        ref_ms: MultiSegment,
        lam_mode: str = "prob",
        w: pd.Series | None = None,
        aff_mode: str = "area",
        starting_k: int = 2,
        min_k_inc: int = 0,
    ):
        self.reference_ms = ref_ms
        self.lam_mode = lam_mode
        self.lam_w = w
        self.aff_mode = aff_mode
        self.starting_k = starting_k
        self.min_k_inc = min_k_inc

    def __call__(self, bh: BoundaryHierarchy) -> MultiSegment:
        lam = self.reference_ms.lam(strategy=self.lam_mode, w=self.lam_w)
        return lam.decode(
            bh, aff_mode=self.aff_mode, starting_k=self.starting_k, min_k_inc=self.min_k_inc
        )


# endregion: Labeling strategies

# region: Helper functions


def bs2uv(bs, min_dur=0.1):
    """Convert a set of boundaries to a set of (u,v) coordinates.
    also return the area of each associated sample point.

    Add a filtering option to remove very small intervals.
    """
    # get rid of bs that are too close to the previous one
    filtered_bs = [bs[0]]
    for b in bs[1:]:
        if b - filtered_bs[-1] > min_dur:
            filtered_bs.append(b)

    # Ensure the last boundary is included
    if filtered_bs[-1] < bs[-1]:
        # If the last added boundary is too close to the end, replace it,
        # unless it is the start boundary itself (to preserve at least one interval)
        if (len(filtered_bs) > 1) and (bs[-1] - filtered_bs[-1] < min_dur):
            filtered_bs[-1] = bs[-1]
        else:
            filtered_bs.append(bs[-1])

    filtered_bs = np.array(filtered_bs)
    print("Filtered boundaries:", len(filtered_bs))

    # get the mid points of the filtered bs
    mid_points = (filtered_bs[:-1] + filtered_bs[1:]) / 2
    n = len(mid_points)
    # Get indices for the upper triangle of an (n x n) matrix
    i, j = np.triu_indices(n)
    dur = filtered_bs[1:] - filtered_bs[:-1]
    area = dur[i] * dur[j]
    # Directly create the (u, v) pairs
    return np.stack([mid_points[i], mid_points[j]], axis=1), area


def build_time_grid(span: TimeSpan, frame_size: float = 0.1) -> np.ndarray:
    """
    Build a grid of times using the same logic as mir_eval to build the ticks
    """
    # Figure out how many frames we need by using `mir_eval`'s exact frame finding logic.
    n_frames = int(
        (_round(span.end.time, frame_size) - _round(span.start.time, frame_size)) / frame_size
    )

    # preserve the start and end times. round always rounds down.
    ts = np.arange(n_frames + 1) * frame_size + span.start.time
    if ts[-1] != span.end.time:
        ts = np.append(ts, span.end.time)
    return ts


def bpc2bs(
    bpc: pd.Series, start_time: float, end_time: float, low_threshold: float = 1e-2
) -> list[RatedBoundary]:
    peak_indices = scipy.signal.find_peaks(bpc)[0]
    peak_times = bpc.index[peak_indices]
    peak_saliences = bpc.iloc[peak_indices]
    max_salience = np.max(peak_saliences) if peak_saliences.size > 0 else 1
    inner_boundaries = [
        RatedBoundary(t, s)
        for t, s in zip(peak_times, peak_saliences, strict=True)
        if s > max_salience * low_threshold
    ]
    final_boundaries = [
        RatedBoundary(start_time, max_salience),
        *inner_boundaries,
        RatedBoundary(end_time, max_salience),
    ]
    return sorted(final_boundaries)


def combine_ms(
    named_ms: dict[str, MultiSegment],
    new_name: str = "combined",
    prune=True,
    reorder=False,
    ignore_names=(),
) -> MultiSegment:
    """
    Combined multiple MS in a single MS, and rename the layers to include the name of the old MS.
    """
    layers = []
    for name, old_ms in named_ms.items():
        if name not in ignore_names:
            for old_layer in old_ms:
                layers.append(replace(old_layer, name=name + ":" + old_layer.name))

    if reorder:
        layers = sorted(layers, key=lambda layer: len(layer))
    new_ms = MultiSegment(layers, name=new_name)
    if prune:
        return new_ms.prune_layers(relabel=False)
    return new_ms


def common_itvls(layers: MultiSegment | list[Segment]) -> np.ndarray:
    common_bs = set()
    for layer in layers:
        common_bs.update(layer.btimes)
    return np.array(sorted(common_bs))
