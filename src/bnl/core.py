"""Core data structures for boundaries-and-labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Any

import jams
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType

__all__ = [
    "Boundary",
    "RatedBoundary",
    "LeveledBoundary",
    "TimeSpan",
    "Segment",
    "BoundaryContour",
    "MultiSegment",
    "Hierarchy",
]


def _validate_time(time: int | float | np.number) -> float:
    """Validates and rounds a time value."""
    if not isinstance(time, int | float | np.number):
        raise TypeError(f"Time must be a number, not {type(time).__name__}.")
    if time < 0:
        raise ValueError("Time cannot be negative.")
    return float(np.round(time, 4))


# region: Point-like Objects


@dataclass(frozen=True, order=True)
class Boundary:
    """A basic, labeled marker on a timeline."""

    time: float = field(default=0.0, compare=True)
    label: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _validate_time(self.time))
        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True, order=True)
class RatedBoundary(Boundary):
    """A boundary with a continuous measure of importance."""

    salience: float = field(default=0.0, compare=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.salience, int | float | np.number):
            raise TypeError(f"Salience must be a number, not {type(self.salience).__name__}.")
        object.__setattr__(self, "salience", float(self.salience))


@dataclass(frozen=True, order=True)
class LeveledBoundary(RatedBoundary):
    """A boundary placed within a formal hierarchy, with labels for its entire lineage."""

    ancestry: list[str] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        # Per spec, salience is determined by level, overriding any passed value.
        object.__setattr__(self, "salience", float(self.level))
        super().__post_init__()

    @property
    def level(self) -> int:
        """The hierarchical level of the boundary, derived from its ancestry."""
        return len(self.ancestry)


# endregion

# region: Span-like Objects (Containers)


@dataclass(frozen=True)
class TimeSpan:
    """The abstract concept of a time interval."""

    start: Boundary
    duration: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", _validate_time(self.duration))

    @property
    def end(self) -> Boundary:
        """The end boundary of the time span."""
        return Boundary(self.start.time + self.duration)


@dataclass(frozen=True)
class Segment(TimeSpan):
    """A simple, ordered sequence of boundaries."""

    boundaries: tuple[Boundary, ...]

    def __init__(self, boundaries: Sequence[Boundary]):
        if len(boundaries) < 2:
            raise ValueError("Segment requires at least two boundaries.")

        sorted_boundaries = tuple(sorted(boundaries))
        start = sorted_boundaries[0]
        duration = sorted_boundaries[-1].time - start.time

        object.__setattr__(self, "boundaries", sorted_boundaries)
        super().__init__(start=start, duration=duration)

    @property
    def segments(self) -> tuple[TimeSpan, ...]:
        """The time spans between adjacent boundaries."""
        return tuple(
            TimeSpan(
                start=self.boundaries[i],
                duration=self.boundaries[i + 1].time - self.boundaries[i].time,
            )
            for i in range(len(self.boundaries) - 1)
        )

    @classmethod
    def from_jams(cls, anno: jams.Annotation) -> Segment:
        """Creates a Segment from a JAMS annotation."""
        boundary_points = {obs.time: obs.value for obs in anno.data}
        if anno.data:
            last_obs = anno.data[-1]
            boundary_points[last_obs.time + last_obs.duration] = None

        boundaries = [Boundary(time, label) for time, label in sorted(boundary_points.items())]
        return cls(boundaries=boundaries)

    @classmethod
    def from_intervals(
        cls, intervals: Sequence[Sequence[float]], labels: Sequence[str | None] | None = None
    ) -> Segment:
        """Creates a Segment from a sequence of [start, end] intervals."""
        if not intervals:
            raise ValueError("Cannot create Segment from empty intervals.")

        # Using a dictionary to ensure unique time points for boundaries
        boundary_points: dict[float, str | None] = {}

        # Create a mapping from start_time to label
        label_map = {}
        if labels:
            for i, interval in enumerate(intervals):
                if i < len(labels):
                    if len(interval) == 2 and interval[0] < interval[1]:
                        label_map[interval[0]] = labels[i]

        for interval in intervals:
            if len(interval) != 2 or interval[0] >= interval[1]:
                continue
            start, end = interval
            # Add start boundary with its label (if any)
            if start not in boundary_points:
                boundary_points[start] = label_map.get(start)
            # Add end boundary, without overwriting a labeled start boundary
            if end not in boundary_points:
                boundary_points[end] = None

        if len(boundary_points) < 2:
            raise ValueError("Intervals must define at least two unique time points.")

        boundaries = [Boundary(time, label) for time, label in boundary_points.items()]
        return cls(boundaries=boundaries)


@dataclass(frozen=True)
class BoundaryContour(TimeSpan):
    """An ordered sequence of rated boundaries, representing a profile of salience over time."""

    boundaries: tuple[RatedBoundary, ...]

    def __init__(self, boundaries: Sequence[RatedBoundary]):
        if len(boundaries) < 2:
            raise ValueError("BoundaryContour requires at least two boundaries.")

        sorted_boundaries = tuple(sorted(boundaries))
        start = sorted_boundaries[0]
        duration = sorted_boundaries[-1].time - start.time

        object.__setattr__(self, "boundaries", sorted_boundaries)
        super().__init__(start=start, duration=duration)


@dataclass(frozen=True)
class MultiSegment(TimeSpan):
    """A container for a collection of different Segment layers."""

    layers: tuple[Segment, ...]

    def __init__(self, layers: Sequence[Segment]):
        if not layers:
            raise ValueError("MultiSegment cannot be empty.")

        all_boundaries = [b for seg in layers for b in seg.boundaries]
        if not all_boundaries:
            raise ValueError("MultiSegment layers cannot be empty of boundaries.")

        start_time = min(b.time for b in all_boundaries)
        end_time = max(b.time for b in all_boundaries)

        object.__setattr__(self, "layers", tuple(layers))
        super().__init__(start=Boundary(start_time), duration=end_time - start_time)

    @classmethod
    def from_jams(cls, anno: jams.Annotation) -> MultiSegment:
        """Creates a `MultiSegment` from a JAMS multi-segment annotation."""
        if anno.namespace != "multi_segment":
            raise ValueError(f"Expected 'multi_segment' namespace, but got '{anno.namespace}'.")

        # Filter out observations that don't have a valid level and label
        valid_obs = [
            obs for obs in anno.data if isinstance(obs.value, dict) and "level" in obs.value and "label" in obs.value
        ]

        if not valid_obs:
            raise ValueError("No valid segments could be created from the JAMS annotation.")

        # Group data by level
        grouped_data = groupby(
            sorted(valid_obs, key=lambda obs: obs.value["level"]),
            key=lambda obs: obs.value["level"],
        )

        layers = []
        for _, group in grouped_data:
            group_list = list(group)
            boundary_points = {obs.time: obs.value["label"] for obs in group_list}
            if group_list:
                last_obs = group_list[-1]
                boundary_points[last_obs.time + last_obs.duration] = None

            boundaries = [Boundary(time, label) for time, label in sorted(boundary_points.items())]
            if len(boundaries) >= 2:
                layers.append(Segment(boundaries=boundaries))
        try:
            return cls(layers=layers)
        except ValueError as e:
            raise ValueError("No valid segments could be created from the JAMS data.") from e

    @classmethod
    def from_json(cls, data: list) -> MultiSegment:
        """
        Creates a `MultiSegment` from a JSON-like structure.
        e.g., a list of layers, where each layer is `[intervals, labels]`.
        """
        layers = []
        for level_data in data:
            try:
                intervals, labels = level_data
            except ValueError:
                continue  # Skip malformed layers

            if len(intervals) != len(labels):
                raise ValueError("Intervals and labels must have the same length.")

            boundaries = []
            for i, (start, end) in enumerate(intervals):
                # each start of the interval is a boundary
                boundaries.append(Boundary(start, labels[i]))
                # Get the end if it's the last one.
                if i == len(intervals) - 1:
                    boundaries.append(Boundary(end, None))

            layers.append(Segment(boundaries=boundaries))
        try:
            return cls(layers=layers)
        except ValueError as e:
            raise ValueError("No valid segments could be created from the JSON data.") from e

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """
        Plots all layers of the MultiSegment on a single axis, stacked vertically.

        Args:
            ax: Matplotlib axes to plot on. If None, a new figure is created.
            **kwargs: Additional keyword arguments to customize the plot.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        if ax is None:
            figsize = kwargs.pop("figsize", (12, 1 + 0.5 * len(self.layers)))
            _, ax = plt.subplots(figsize=figsize)

        # Build a consistent color map for all unique labels across all layers
        all_labels = {s.start.label for layer in self.layers for s in layer.segments if s.start.label}
        palette = list(mcolors.TABLEAU_COLORS.values())
        color_map: dict[str, ColorType] = {
            label: palette[i % len(palette)] for i, label in enumerate(sorted(all_labels))
        }

        bar_height = kwargs.pop("bar_height", 0.8)

        # Plot layers from bottom (index 0) to top
        for i, layer in enumerate(self.layers):
            y_center = i + 0.5
            for span in layer.segments:
                color = color_map.get(span.start.label or str(span.start), "gray")
                ax.barh(
                    y=y_center,
                    width=span.duration,
                    height=bar_height,
                    left=span.start.time,
                    color=color,
                    edgecolor=kwargs.get("edgecolor", "white"),
                    alpha=kwargs.get("alpha", 0.7),
                )
                if span.start.label:
                    ax.text(
                        span.start.time + span.duration * 0.01,
                        y_center,
                        span.start.label,
                        va="center",
                        ha="left",
                        fontsize="small",
                    )

        # Configure the axes
        ax.set_xlim(self.start.time, self.end.time)
        ax.set_ylim(0, len(self.layers))
        ax.set_yticks([i + 0.5 for i in range(len(self.layers))])
        ax.set_yticklabels([f"Layer {i}" for i in range(len(self.layers))])
        ax.set_title(kwargs.get("title", "MultiSegment"))

        return ax

    def to_boundary_contour(self, method: str = "frequency") -> BoundaryContour:
        """
        Converts the MultiSegment into a BoundaryContour by calculating salience.

        This method collapses the layered structure into a single sequence of
        rated boundaries, where the salience of each boundary is determined
        by the specified method.

        Args:
            method: The salience calculation method. Currently, only
                'frequency' is supported.

        Returns:
            A new `BoundaryContour` object.
        """
        from . import ops

        if method == "frequency":
            salience_map = ops.calculate_frequency_salience(self)
        else:
            raise ValueError(f"Unsupported salience method: '{method}'")

        rated_boundaries = [RatedBoundary(time=time, salience=salience) for time, salience in salience_map.items()]
        return BoundaryContour(boundaries=rated_boundaries)


@dataclass(frozen=True)
class Hierarchy(TimeSpan):
    """The definitive, well-formed structural hierarchy."""

    boundaries: tuple[LeveledBoundary, ...]

    def __init__(self, boundaries: Sequence[LeveledBoundary]):
        if len(boundaries) < 2:
            raise ValueError("Hierarchy requires at least two boundaries.")

        sorted_boundaries = tuple(sorted(boundaries))
        start = sorted_boundaries[0]
        duration = sorted_boundaries[-1].time - start.time

        object.__setattr__(self, "boundaries", sorted_boundaries)
        super().__init__(start=start, duration=duration)

    def to_multisegment(self) -> MultiSegment:
        """Converts the hierarchy into its MultiSegment representation."""
        if not self.boundaries:
            # This case should ideally not be hit due to __init__ check.
            return MultiSegment(layers=[])

        max_level = max((b.level for b in self.boundaries), default=0)
        if max_level == 0:
            return MultiSegment(layers=[])

        layers = []
        for level in range(1, max_level + 1):
            # Boundaries for this segment are those from the hierarchy at or above the current level
            level_boundaries = [Boundary(b.time, b.ancestry[level - 1]) for b in self.boundaries if b.level >= level]

            # We only create a segment if there are enough boundaries to form a span.
            # We use a set to get unique time points, as multiple boundaries can exist at the same time.
            if len({b.time for b in level_boundaries}) >= 2:
                layers.append(Segment(boundaries=level_boundaries))

        if not layers:
            return MultiSegment(layers=[])

        return MultiSegment(layers=layers)


# endregion
