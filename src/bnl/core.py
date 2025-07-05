"""Core data structures for boundaries-and-labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    """Validates and rounds a time value."""
    if not isinstance(time, int | float | np.number):
        raise TypeError(f"Time must be a number, not {type(time).__name__}.")
    if time < 0:
        raise ValueError("Time cannot be negative.")
    return float(np.round(time, 4))


@dataclass(frozen=True, order=True)
class Boundary:
    """A time point, optionally with a label."""

    time: float = field(default=0.0, compare=True)
    label: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _validate_time(self.time))
        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class TimeSpan:
    """A labeled time interval."""

    start: Boundary = field(default_factory=lambda: Boundary(0.0, None))
    duration: float = 1.0
    label: str | None = None

    # Adding explicit init to bypass dataclass inheritance issues
    def __init__(self, start: Boundary, duration: float, label: str | None = None):
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "label", label)
        self.__post_init__()

    @property
    def end(self) -> Boundary:
        """The end boundary of the time span."""
        return Boundary(self.start.time + self.duration)

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, but got {self.duration}.")
        if self.label is not None:
            # Ensure label is a string for consistency
            object.__setattr__(self, "label", str(self.label))

    def __str__(self) -> str:
        lab = f": {self.label}" if self.label else ""
        return f"TimeSpan([{self.start.time:.2f}s-{self.end.time:.2f}s], {self.duration:.2f}s{lab})"

    def __repr__(self) -> str:
        return f"TimeSpan(start={self.start!r}, duration={self.duration!r}, label={self.label!r})"


@dataclass(frozen=True)
class Segmentation(TimeSpan):
    """A sequence of contiguous `TimeSpan`s."""

    boundaries: Sequence[Boundary] = field(default_factory=tuple)

    def __init__(
        self,
        start: Boundary,
        duration: float,
        boundaries: Sequence[Boundary],
        label: str | None = None,
    ):
        # Set child attributes FIRST to ensure they exist before post-init validation
        object.__setattr__(self, "boundaries", tuple(boundaries))
        # Now call parent init, which will trigger post-init
        super().__init__(start, duration, label)

    def __post_init__(self) -> None:
        # Note: super().__post_init__() is not needed because the parent init calls it
        if not self.boundaries:
            raise ValueError("Segmentation requires at least one boundary.")

        # Ensure boundaries are sorted
        sorted_boundaries = tuple(sorted(self.boundaries))
        object.__setattr__(self, "boundaries", sorted_boundaries)

        # Check consistency
        if not np.isclose(self.start.time, sorted_boundaries[0].time):
            raise ValueError(
                f"Segmentation start time {self.start.time} "
                f"does not match first boundary time {sorted_boundaries[0].time}."
            )

        if not np.isclose(self.end.time, sorted_boundaries[-1].time):
            raise ValueError(
                f"Segmentation end time {self.end.time} does not match last boundary time {sorted_boundaries[-1].time}."
            )

    @property
    def segments(self) -> tuple[TimeSpan, ...]:
        """The segments in the segmentation."""
        return tuple(
            TimeSpan(
                start=self.boundaries[i],
                duration=self.boundaries[i + 1].time - self.boundaries[i].time,
                label=self.boundaries[i].label,
            )
            for i in range(len(self.boundaries) - 1)
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx]

    def __repr__(self) -> str:
        label_str = f"label='{self.label}', " if self.label else ""
        return f"Segmentation({label_str}{len(self)} segments, duration={self.duration:.2f}s)"

    @classmethod
    def from_boundaries(
        cls, times: Sequence[float], segment_labels: Sequence[str | None] | None = None, label: str | None = None
    ) -> Segmentation:
        """Creates a `Segmentation` from boundary times."""
        if not times:
            raise ValueError("Cannot create Segmentation from empty times.")

        sorted_times = sorted(list(set(times)))
        if segment_labels is None:
            segment_labels = [None] * (len(sorted_times) - 1)

        # Each boundary gets the label of the segment it starts
        boundaries = []
        for i, t in enumerate(sorted_times):
            boundary_label = segment_labels[i] if i < len(segment_labels) else None
            boundaries.append(Boundary(t, label=boundary_label))

        start_boundary = boundaries[0]
        duration = boundaries[-1].time - start_boundary.time
        return cls(start=start_boundary, duration=duration, boundaries=boundaries, label=label)

    @classmethod
    def from_intervals(
        cls,
        intervals: Sequence[Sequence[float]],
        segment_labels: Sequence[str | None] | None = None,
        label: str | None = None,
    ) -> Segmentation:
        """Creates a `Segmentation` from (start, end) intervals."""
        if len(intervals) == 0:
            raise ValueError("Cannot create Segmentation from empty intervals.")

        boundary_times = sorted(list(set(t for i in intervals for t in i)))
        return cls.from_boundaries(boundary_times, segment_labels=segment_labels, label=label)

    @classmethod
    def from_jams(
        cls,
        anno: jams.Annotation,
        label: str | None = None,
        start_time: float | None = None,
        duration: float | None = None,
    ) -> Segmentation:
        """
        Creates a `Segmentation` from a JAMS annotation.

        Args:
            anno: JAMS annotation to convert.
            label: Optional label for the segmentation.
            start_time: Optional start time to enforce.
            duration: Optional duration to enforce.

        Returns:
            A new `Segmentation` object.
        """
        boundary_dict = {}

        for obs in anno:
            time, obs_duration, value = obs.time, obs.duration, obs.value
            obs_label = value.get("label") if isinstance(value, dict) else value

            # Prioritize labeled boundaries. If a boundary at this time already exists,
            # only overwrite it if the new one has a non-None label.
            if time not in boundary_dict or obs_label is not None:
                boundary_dict[time] = obs_label

            if obs_duration > 0:
                end_time = time + obs_duration
                # Add the end boundary only if it doesn't already exist.
                if end_time not in boundary_dict:
                    boundary_dict[end_time] = None

        # If a specific time range is given, enforce it by adding/overwriting boundaries.
        if start_time is not None:
            if start_time not in boundary_dict:
                boundary_dict[start_time] = None
            if duration is not None:
                end_time = start_time + duration
                if end_time not in boundary_dict:
                    boundary_dict[end_time] = None
        elif duration is not None:
            jams_start = min((obs.time for obs in anno), default=0.0)
            if jams_start not in boundary_dict:
                boundary_dict[jams_start] = None
            end_time = jams_start + duration
            if end_time not in boundary_dict:
                boundary_dict[end_time] = None

        if not boundary_dict:
            if start_time is not None and duration is not None:
                # Create a simple segmentation if there's no data but a range is defined
                boundaries = [Boundary(start_time), Boundary(start_time + duration)]
            else:
                raise ValueError("Cannot create Segmentation from a JAMS annotation with no data and no defined range.")
        else:
            boundaries = [Boundary(t, label) for t, label in boundary_dict.items()]

        unique_boundaries = sorted(boundaries)
        final_start_boundary = unique_boundaries[0]
        final_duration = unique_boundaries[-1].time - final_start_boundary.time

        # The overall segmentation label is passed in, or defaults to the annotation's namespace.
        # The labels on individual boundaries are for the segments they start.
        return cls(
            start=final_start_boundary,
            duration=final_duration,
            boundaries=unique_boundaries,
            label=label or anno.namespace,
        )


@dataclass(frozen=True)
class Hierarchy(TimeSpan):
    """A hierarchical structure of `Segmentation`s."""

    layers: Sequence[Segmentation] = field(default_factory=tuple)

    def __init__(
        self,
        start: Boundary,
        duration: float,
        layers: Sequence[Segmentation],
        label: str | None = None,
    ):
        # Set child attributes FIRST
        object.__setattr__(self, "layers", tuple(layers))
        # Now call parent init, which triggers post-init
        super().__init__(start, duration, label)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Hierarchy must contain at least one layer.")

        # Ensure layers are a tuple
        object.__setattr__(self, "layers", tuple(self.layers))

        if len(self.layers) > 1:
            for i, layer in enumerate(self.layers):
                if not (np.isclose(layer.start.time, self.start.time) and np.isclose(layer.end.time, self.end.time)):
                    raise ValueError(
                        f"All layers must span the same time range. Layer {i} "
                        f"({layer.start.time:.2f}-{layer.end.time:.2f}) does not match "
                        f"Hierarchy ({self.start.time:.2f}-{self.end.time:.2f})."
                    )

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx]

    def __repr__(self) -> str:
        label_str = f"label='{self.label}'" if self.label else ""
        return f"Hierarchy({label_str}, {len(self)} layers, duration={self.duration:.2f}s)"

    @classmethod
    def from_jams(cls, anno: jams.Annotation, label: str | None = None) -> Hierarchy:
        """Creates a `Hierarchy` from a JAMS multi-segment annotation."""
        from collections import defaultdict

        if anno.namespace != "multi_segment":
            raise ValueError(f"Expected 'multi_segment' namespace, but got '{anno.namespace}'.")

        # Group observations by level
        layers_data = defaultdict(list)
        for obs in anno.data:
            # The "value" of a multi-segment observation contains the sub-segment's metadata
            level = obs.value.get("level")
            if level is not None:
                layers_data[level].append(obs)

        if not layers_data:
            raise ValueError("Cannot create a Hierarchy from a JAMS annotation with no levels.")

        # Determine the overall time range for the hierarchy.
        # It's defined by the earliest start time and the specified annotation duration.
        start_time = min((obs.time for obs in anno), default=0.0)
        duration = anno.duration

        if duration is None:
            # Fallback if the parent annotation has no duration set
            all_times = {
                t
                for l_obs in layers_data.values()
                for obs in l_obs
                for t in [obs.time, obs.time + obs.duration]
                if obs.duration is not None
            }
            if not all_times:
                # Fallback to just observation times if no durations are present
                all_times = {obs.time for l_obs in layers_data.values() for obs in l_obs}

            if not all_times:
                raise ValueError("Cannot determine time range from JAMS data with no observations or durations.")

            end_time = max(all_times)
            duration = end_time - start_time

        # Create a segmentation for each level, ensuring it spans the full hierarchy duration
        layers = []
        for level in sorted(layers_data.keys()):
            level_obs = layers_data[level]

            # Create a temporary annotation for this layer's data
            sub_anno = jams.Annotation(
                namespace="segment",  # A generic namespace for the temporary annotation
                data=level_obs,
                sandbox=anno.sandbox,
                annotation_metadata=anno.annotation_metadata,
            )

            # Create the segmentation, forcing it to align with the parent hierarchy's time range
            segmentation = Segmentation.from_jams(
                sub_anno,
                label=f"level_{level}",
                start_time=start_time,
                duration=duration,
            )
            layers.append(segmentation)

        return cls(
            start=Boundary(start_time),
            duration=duration,
            layers=layers,
            label=label or anno.namespace,
        )

    @classmethod
    def from_json(cls, data: list, label: str | None = None) -> Hierarchy:
        """
        Creates a `Hierarchy` from a JSON-like structure.
        e.g., a list of layers, where each layer is a list of [intervals, labels].
        """
        layers = []
        all_boundaries: set[float] = set()

        for i, (intervals, labels) in enumerate(data):
            seg = Segmentation.from_intervals(intervals, segment_labels=labels, label=f"level_{i}")
            layers.append(seg)
            all_boundaries.update(b.time for b in seg.boundaries)

        if not layers:
            raise ValueError("Cannot create a Hierarchy from empty JSON data.")

        start_time = min(all_boundaries) if all_boundaries else 0
        end_time = max(all_boundaries) if all_boundaries else 0
        duration = end_time - start_time

        return cls(start=Boundary(start_time), duration=duration, layers=layers, label=label)


@dataclass(order=True, frozen=True)
class RatedBoundary:
    """A boundary with a salience level."""

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
    """An intermediate collection of rated boundaries for synthesis."""

    events: Sequence[RatedBoundary]
    start_time: float
    end_time: float

    def group_boundaries(self, strategy: strategies.BoundaryGroupingStrategy) -> RatedBoundaries:
        """Groups close boundaries using a given strategy."""
        grouped_events = strategy.group(self.events)
        return RatedBoundaries(events=grouped_events, start_time=self.start_time, end_time=self.end_time)

    def quantize_level(self, strategy: strategies.LevelGroupingStrategy) -> ProperHierarchy:
        """Converts saliences to levels and synthesizes a `ProperHierarchy`."""
        return strategy.quantize(self)


@dataclass(frozen=True)
class ProperHierarchy(Hierarchy):
    """A `Hierarchy` with monotonically nested layers."""

    def __post_init__(self) -> None:
        # First, run the parent's post-init
        super().__post_init__()

        # Then, validate the monotonic property
        # Layer 0 is coarsest, layer i+1 is finer than layer i
        for i in range(len(self.layers) - 1):
            coarser_layer_boundaries = {b.time for b in self.layers[i].boundaries}
            finer_layer_boundaries = {b.time for b in self.layers[i + 1].boundaries}
            if not coarser_layer_boundaries.issubset(finer_layer_boundaries):
                raise ValueError(
                    f"Monotonicity violation: Layer {i} has boundaries not present in the finer layer {i + 1}."
                )

    @staticmethod
    def from_rated_boundaries(
        events: Sequence[RatedBoundary], start_time: float, end_time: float, label: str | None = None
    ) -> ProperHierarchy:
        """Synthesizes a `ProperHierarchy` from rated boundaries."""
        from .strategies import DirectSynthesisStrategy

        # Wrap in RatedBoundaries to use the strategy
        rated_boundaries = RatedBoundaries(events=events, start_time=start_time, end_time=end_time)

        # Use the strategy to perform the synthesis
        strategy = DirectSynthesisStrategy()
        return strategy.quantize(rated_boundaries)
