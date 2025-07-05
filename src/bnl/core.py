"""Core data structures for boundaries-and-labels, a boundary-centric paradigm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jams
import numpy as np

if TYPE_CHECKING:
    from . import strategies

__all__ = [
    "Boundary",
    "TimeSpan",
    "Segmentation",
    "Hierarchy",
    "RatedBoundary",
    "RatedBoundaries",
    "ProperHierarchy",
]


def _validate_time(time: int | float | np.number) -> float:
    """Validate and round a time value."""
    if not isinstance(time, int | float | np.number):
        raise TypeError(f"Time must be a number, not {type(time).__name__}.")
    if time < 0:
        raise ValueError("Time cannot be negative.")
    return float(np.round(time, 4))


@dataclass(frozen=True, order=True)
class Boundary:
    """A time point, optionally with a label. The fundamental unit."""

    time: float = field(default=0.0, compare=True)
    label: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _validate_time(self.time))
        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class TimeSpan:
    """A labeled time span defined by a start boundary and a duration."""

    start: Boundary = field(default_factory=Boundary)
    duration: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, but got {self.duration}.")
        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))

    @property
    def end(self) -> Boundary:
        """The end boundary of the time span (calculated)."""
        return Boundary(time=self.start.time + self.duration)

    def __str__(self) -> str:
        lab = f": {self.label}" if self.label else ""
        return f"TimeSpan([{self.start.time:.2f}s-{self.end.time:.2f}s], {self.duration:.2f}s{lab})"

    def __repr__(self) -> str:
        return f"TimeSpan(start={self.start!r}, duration={self.duration!r}, label={self.label!r})"


@dataclass(frozen=True)
class Segmentation:
    """A contiguous sequence of TimeSpans defined by an ordered collection of boundaries and labels."""

    boundaries: Sequence[Boundary]
    labels: Sequence[str | None]
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.boundaries:
            raise ValueError("Segmentation requires at least one boundary.")
        if len(self.labels) != len(self.boundaries) - 1:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must be one less than "
                f"the number of boundaries ({len(self.boundaries)})."
            )

        # Ensure boundaries are sorted
        sorted_boundaries = tuple(sorted(self.boundaries))
        object.__setattr__(self, "boundaries", sorted_boundaries)

        # Ensure labels are a tuple
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def start(self) -> Boundary:
        return self.boundaries[0]

    @property
    def end(self) -> Boundary:
        return self.boundaries[-1]

    @property
    def duration(self) -> float:
        return self.end.time - self.start.time

    @property
    def segments(self) -> tuple[TimeSpan, ...]:
        """The segments composing the segmentation."""
        return tuple(
            TimeSpan(
                start=self.boundaries[i],
                duration=self.boundaries[i + 1].time - self.boundaries[i].time,
                label=self.labels[i],
            )
            for i in range(len(self.labels))
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx]

    def __repr__(self) -> str:
        name_str = f"name='{self.name}', " if self.name else ""
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

    @classmethod
    def from_boundaries(
        cls, boundaries: Sequence[float], labels: Sequence[str | None] | None = None, name: str | None = None
    ) -> Segmentation:
        """Creates a Segmentation from a sequence of boundary times."""
        if labels is None:
            labels = [None] * (len(boundaries) - 1)
        b_objs = [Boundary(t) for t in sorted(list(set(boundaries)))]
        return cls(boundaries=b_objs, labels=labels, name=name)

    @classmethod
    def from_intervals(
        cls, intervals: np.ndarray, labels: list[str | None] | None = None, name: str | None = None
    ) -> Segmentation:
        """Creates a Segmentation from a numpy array of intervals."""
        intervals_np = np.asanyarray(intervals)
        boundary_times = np.unique(intervals_np.flatten())
        return cls.from_boundaries(list(boundary_times.astype(float)), labels=labels, name=name)

    @classmethod
    def from_jams(cls, anno: jams.Annotation) -> Segmentation:
        # Event-like annotations (e.g., beats) have duration 0
        is_event_like = all(obs.duration == 0 for obs in anno)

        sorted_observations = sorted(anno, key=lambda o: o.time)

        if is_event_like:
            boundaries = [Boundary(time=obs.time, label=str(obs.value)) for obs in sorted_observations]
            labels = [None] * (len(boundaries) - 1)
        else:
            times = sorted(
                {obs.time for obs in sorted_observations} | {obs.time + obs.duration for obs in sorted_observations}
            )
            boundaries = [Boundary(time=t) for t in times]
            labels = [obs.value for obs in sorted_observations]

        return cls(boundaries=boundaries, labels=labels, name=anno.namespace)


@dataclass(frozen=True)
class Hierarchy:
    """A hierarchical structure of segmentations, aligned in time."""

    layers: Sequence[Segmentation]
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Hierarchy must contain at least one layer.")

        # Ensure layers are a tuple
        object.__setattr__(self, "layers", tuple(self.layers))

        if len(self.layers) > 1:
            start_time = self.layers[0].start.time
            end_time = self.layers[0].end.time
            for i, layer in enumerate(self.layers[1:], 1):
                if not (np.isclose(layer.start.time, start_time) and np.isclose(layer.end.time, end_time)):
                    raise ValueError(
                        f"All layers must span the same time range. Layer {i} "
                        f"({layer.start.time:.2f}-{layer.end.time:.2f}) does not match "
                        f"Layer 0 ({start_time:.2f}-{end_time:.2f})."
                    )

    @property
    def start(self) -> Boundary:
        return self.layers[0].start

    @property
    def end(self) -> Boundary:
        return self.layers[0].end

    @property
    def duration(self) -> float:
        return self.end.time - self.start.time

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx]

    def __repr__(self) -> str:
        name_str = f"name='{self.name}'" if self.name else ""
        return f"Hierarchy({name_str}, {len(self)} layers, duration={self.duration:.2f}s)"

    @classmethod
    def from_jams(cls, jams_annotation: jams.Annotation, name: str | None = None) -> Hierarchy:
        if jams_annotation.namespace != "multi_segment":
            raise ValueError(f"Expected 'multi_segment' namespace, got '{jams_annotation.namespace}'")

        from jams.eval import hierarchy_flatten

        hier_intervals, hier_labels = hierarchy_flatten(jams_annotation)

        segmentations = []
        for i, (intervals, labels) in enumerate(zip(hier_intervals, hier_labels)):
            seg = Segmentation.from_intervals(intervals=np.array(intervals), labels=list(labels), name=f"level_{i}")
            segmentations.append(seg)

        final_name = name
        if not final_name:
            annotator_meta = jams_annotation.annotation_metadata.annotator
            if annotator_meta and "name" in annotator_meta:
                final_name = annotator_meta["name"]

        return cls(layers=segmentations, name=final_name)

    @classmethod
    def from_json(cls, json_data: list[list[list[Any]]], name: str | None = None) -> Hierarchy:
        segmentations = []
        for i, layer_data in enumerate(json_data):
            if not (isinstance(layer_data, list) and len(layer_data) == 2):
                raise ValueError(f"Layer {i} malformed. Expected [intervals, labels].")

            intervals_data, labels_data = layer_data
            intervals_np = np.array([item[0] if len(item) == 1 else item for item in intervals_data])

            seg = Segmentation.from_intervals(intervals=intervals_np, labels=labels_data, name=f"layer_{i}")
            segmentations.append(seg)

        return cls(layers=segmentations, name=name)


@dataclass(order=True, frozen=True)
class RatedBoundary:
    """A boundary with an associated salience level."""

    time: float
    salience: float | int
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _validate_time(self.time))
        if not isinstance(self.salience, int | float):
            raise TypeError(f"Salience must be a number, not {type(self.salience).__name__}.")
        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))

    def __repr__(self) -> str:
        return f"RatedBoundary(time={self.time:.2f}, salience={self.salience:.2f}, label={self.label!r})"


@dataclass(frozen=True)
class RatedBoundaries:
    """
    An intermediate collection of rated boundaries, ready for synthesis.

    This container supports a fluent, chainable API for multi-stage
    processing, such as grouping and quantization. It holds the boundaries
    (events), along with the start and end times of the original analysis frame.
    """

    events: Sequence[RatedBoundary]
    start_time: float
    end_time: float

    def group_boundaries(self, strategy: strategies.BoundaryGroupingStrategy) -> RatedBoundaries:
        """Groups boundaries that are close in time using a given strategy."""
        grouped_events = strategy.group(self.events)
        return RatedBoundaries(events=grouped_events, start_time=self.start_time, end_time=self.end_time)

    def quantize_level(self, strategy: strategies.LevelGroupingStrategy) -> ProperHierarchy:
        """Converts saliences to levels and synthesizes a ProperHierarchy."""
        return strategy.quantize(self)


@dataclass(frozen=True)
class ProperHierarchy(Hierarchy):
    """
    A `Hierarchy` guaranteed to have monotonically nested layers.

    This class enforces that each layer's boundaries are a strict subset of the
    boundaries in the layer below it, which is a key requirement for many
    music structure analysis algorithms.
    """

    def __post_init__(self) -> None:
        # First, run the parent's post-init
        super().__post_init__()

        # Then, validate the monotonic property
        for i in range(len(self.layers) - 1):
            finer_layer_boundaries = {b.time for b in self.layers[i].boundaries}
            coarser_layer_boundaries = {b.time for b in self.layers[i + 1].boundaries}
            if not coarser_layer_boundaries.issubset(finer_layer_boundaries):
                raise ValueError(
                    f"Monotonicity violation: Layer {i + 1} has boundaries not present in the finer layer {i}."
                )

    @staticmethod
    def from_rated_boundaries(
        events: Sequence[RatedBoundary], start_time: float, end_time: float, name: str | None = None
    ) -> ProperHierarchy:
        """
        Synthesizes a ProperHierarchy from a sequence of rated boundaries.

        Saliences are converted to integer depths, and layers are built such
        that higher-salienced events appear in deeper layers.
        """
        from .strategies import DirectSynthesisStrategy

        # Wrap in RatedBoundaries to use the strategy
        rated_boundaries = RatedBoundaries(events=events, start_time=start_time, end_time=end_time)

        # Use the strategy to perform the synthesis
        strategy = DirectSynthesisStrategy()
        return strategy.quantize(rated_boundaries)
