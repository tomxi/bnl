"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import jams
from matplotlib.axes import Axes

# region: Point-like Objects


@dataclass(frozen=True, order=True)
class Boundary:
    """
    A divider in the flow of time in seconds, quantized to 1e-5 seconds.
    """

    time: float

    def __post_init__(self) -> None:
        rounded_time = round(self.time, 5)
        object.__setattr__(self, "time", rounded_time)

    def __repr__(self) -> str:
        return f"B({self.time})"


@dataclass(frozen=True, order=True)
class RatedBoundary(Boundary):
    """
    A boundary with a continuous measure of importance or salience.
    """

    salience: float

    def __repr__(self) -> str:
        return f"RB({self.time}, {self.salience:.2f})"


@dataclass(frozen=True, order=True, init=False)
class LeveledBoundary(RatedBoundary):
    """
    A boundary that exists in a monotonic hierarchy, that exists in the first `level` layers.

    The `level` must be a positive integer, and the `salience` attribute
    is automatically set to be equal to the `level`.
    """

    level: int

    def __init__(self, time: float, level: int):
        """
        Initializes a LeveledBoundary, deriving salience from level.

        Parameters
        ----------
        time : float
            The time of the boundary in seconds.
        level : int
            The discrete hierarchical level of the boundary. Must be a positive integer.
        """
        if not isinstance(level, int) or level <= 0:
            raise ValueError("`level` must be a positive integer.")

        # Manually set the attributes for this frozen instance.
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "salience", float(level))

        # Explicitly call the Boundary object's post-init for time validation.
        super().__post_init__()

    def __repr__(self) -> str:
        return f"LB({self.time}, {self.level})"


# endregion

# region: Span-like Objects (Containers)


@dataclass
class TimeSpan:
    """
    Represents a generic time interval.

    Must have a non-zero, positive duration. Allows empty string for name as default.
    """

    start: Boundary
    end: Boundary
    name: str = ""

    def __post_init__(self) -> None:
        if self.end.time <= self.start.time:
            raise ValueError("TimeSpan must have a non-zero, positive duration.")

    @property
    def duration(self) -> float:
        return self.end.time - self.start.time

    def __repr__(self) -> str:
        return f"TS({self.start}-{self.end}, {self.name})"

    def __str__(self) -> str:
        return self.name if self.name != "" else f"[{self.start.time:.2f}-{self.end.time:.2f}]"

    def plot(self, ax: Axes, **kwargs: Any) -> Axes:
        """
        Plots the time span on a set of axes.

        A wrapper around `bnl.viz.plot_timespan`.
        """
        from . import viz

        return viz.plot_timespan(self, ax=ax, **kwargs)


class Segment(TimeSpan):
    """
    An ordered sequence of boundaries that partition a span into labeled sections.
    Represents one layer of annotation.
    """

    def __init__(self, boundaries: Sequence[Boundary], labels: Sequence[str], name: str = "Segment"):
        """
        Initializes the Segment.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Parameters
        ----------
        boundaries : Sequence[Boundary]
            A list of at least two boundaries, sorted by time.
        labels : Sequence[str]
            A list of labels for the sections. Must be `len(boundaries) - 1`.
        name : str, optional
            Name of the segment. Defaults to "Segment".
        """
        if not boundaries or len(boundaries) < 2:
            raise ValueError("A Segment requires at least two boundaries.")
        if len(labels) != len(boundaries) - 1:
            raise ValueError("Number of labels must be one less than the number of boundaries.")

        self.boundaries = list(boundaries)
        if self.boundaries != sorted(self.boundaries):
            raise ValueError("Boundaries must be sorted by time.")

        self.labels = labels
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    @property
    def sections(self) -> Sequence[TimeSpan]:
        """A list of all the labeled time spans that compose the segment."""
        return [
            TimeSpan(start=self.boundaries[i], end=self.boundaries[i + 1], name=self.labels[i])
            for i in range(len(self.labels))
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, key: int) -> TimeSpan:
        return self.sections[key]

    @classmethod
    def from_jams(cls, segment_annotation: jams.Annotation, name: str = "Segment") -> Segment:
        """
        Data Ingestion from jams format.
        """
        itvls, labels = segment_annotation.to_interval_values()
        return cls.from_itvls(itvls, labels, name=name)

    @classmethod
    def from_itvls(cls, itvls: Sequence[Sequence[float]], labels: Sequence[str], name: str = "Segment") -> Segment:
        """
        Data Ingestion from `mir_eval` format of boundaries and labels.
        """
        boundaries = [Boundary(itvl[0]) for itvl in itvls]  # assume intervals have no overlap or gaps
        boundaries.append(Boundary(itvls[-1][1]))  # tag on the end time of the last interval
        return cls(boundaries=boundaries, labels=labels, name=name)

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """
        Plots the segment on a set of axes.

        A wrapper around `bnl.viz.plot_segment`.
        """
        from . import viz

        return viz.plot_segment(self, ax=ax, **kwargs)


class MultiSegment(TimeSpan):
    """
    The primary input object for analysis, containing multiple Segment layers.
    """

    def __init__(self, layers: Sequence[Segment], name: str = "Hierarchical Segmentation"):
        """
        Initializes the MultiSegment.

        The `start` and `end` attributes are automatically derived from the first layer.

        Parameters
        ----------
        layers : list[Segment]
            A list of Segment layers. All layers must have the same start and end times.
        name : str, optional
            Name of the object. Defaults to "Hierarchical Segmentation".
        """
        if len(layers) <= 0:
            raise ValueError("MultiSegment must contain at least one Segment layer.")

        self.layers = layers

        # All layers must span the same time interval.
        # Use the first layer as the reference for comparison.
        first_layer = layers[0]
        expected_start, expected_end = first_layer.start, first_layer.end

        for layer in layers[1:]:
            if layer.start != expected_start:
                raise ValueError(
                    f"All layers must have the same start time. current: {layer.start} != {expected_start}"
                )
            if layer.end != expected_end:
                raise ValueError(f"All layers must have the same end time. current: {layer.end} != {expected_end}")

        super().__init__(start=expected_start, end=expected_end, name=name)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, key: int) -> Segment:
        return self.layers[key]

    @classmethod
    def from_json(cls, json_data: list, name: str = "JSON Annotation") -> MultiSegment:
        """
        Data Ingestion from adobe json format, list[layers].
        each layer is [itvls, labels], itvls is list[[start_time, end_time]], labels is list[str]
        """
        layers = []
        for i, layer in enumerate(json_data, start=1):
            itvls, labels = layer
            layers.append(Segment.from_itvls(itvls, labels, name=f"L{i:02d}"))
        return cls(layers=layers, name=name)

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """
        Plots the MultiSegment on an axes.

        A wrapper around `bnl.viz.plot_multisegment`.
        """
        from . import viz

        return viz.plot_multisegment(self, ax=ax, **kwargs)

    def to_contour(self, strategy: str = "depth") -> BoundaryContour:
        """
        Calculates boundary salience and converts to a BoundaryContour.

        This is a convenience wrapper around `bnl.ops.boundary_salience`.

        Parameters
        ----------
        strategy : {'depth', 'count', 'prob'}, default 'depth'
            The salience calculation strategy to use. See `bnl.ops.boundary_salience`
            for more details.

        Returns
        -------
        BoundaryContour
            The resulting boundary structure.
        """
        from . import ops  # Local import to avoid circular dependency at runtime

        return ops.boundary_salience(self, strategy=strategy)

    @staticmethod
    def align_layers(layers: Sequence[Segment]) -> Sequence[Segment]:
        """
        Adjusts a list of Segment layers to have a common start and end time.

        This is achieved by finding the earliest start time and latest end time
        among all layers, and then extending each layer to this common span.
        The first and last sections of each layer are stretched to cover the new span.
        This method returns a new list of aligned Segment objects.
        """
        if not layers:
            return []

        min_start_time = min(layer.start.time for layer in layers)
        max_end_time = max(layer.end.time for layer in layers)

        aligned_layers = []
        for layer in layers:
            new_boundaries = layer.boundaries.copy()
            new_boundaries[0] = replace(new_boundaries[0], time=min_start_time)
            new_boundaries[-1] = replace(new_boundaries[-1], time=max_end_time)

            aligned_layers.append(Segment(boundaries=new_boundaries, labels=layer.labels, name=layer.name))

        return aligned_layers


class BoundaryContour(TimeSpan):
    """
    An intermediate, purely structural representation of boundary salience over time.
    """

    def __init__(self, name: str, boundaries: Sequence[RatedBoundary]):
        """
        Initializes the BoundaryContour.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Parameters
        ----------
        name : str
            Name of the contour.
        boundaries : Sequence[RatedBoundary]
            A list of rated boundaries. They will be sorted by time upon initialization.
        """
        if len(boundaries) < 2:
            raise ValueError("At least 2 boundaries for a TimeSpan!")
        self.boundaries: Sequence[RatedBoundary] = sorted(boundaries)
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    def __len__(self) -> int:
        return len(self.boundaries)

    def __getitem__(self, key: int) -> RatedBoundary:
        return self.boundaries[key]

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """
        Plots the BoundaryContour on an axes.

        A wrapper around `bnl.viz.plot_boundary_contour`.
        """
        from . import viz

        return viz.plot_boundary_contour(self, ax=ax, **kwargs)

    def clean(self, strategy: str = "absorb", **kwargs) -> BoundaryContour:
        """
        Cleans up the boundary contour using a specified strategy.

        This is a convenience wrapper around `bnl.ops.clean_boundaries`.

        Parameters
        ----------
        strategy : {'absorb', 'kde'}, default 'absorb'
            The cleaning strategy to use. See `bnl.ops.clean_boundaries`
            for more details and strategy-specific parameters.
        **kwargs
            Additional keyword arguments to pass to the strategy (e.g., `window`).

        Returns
        -------
        BoundaryContour
            A new, cleaned BoundaryContour.
        """
        from . import ops

        return ops.clean_boundaries(self, strategy=strategy, **kwargs)

    def to_hierarchy(self) -> BoundaryHierarchy:
        """
        [STUB] Converts the BoundaryContour to a BoundaryHierarchy by setting boundary salience to discrete levels.
        """
        from . import ops

        # TODO: Implement this.
        return ops.level_by_distinct_salience(self)


class BoundaryHierarchy(BoundaryContour):
    """
    The structural output of the monotonic casting process.
    """

    boundaries: Sequence[LeveledBoundary]

    def __init__(self, name: str, boundaries: Sequence[LeveledBoundary]):
        """
        Initializes the BoundaryHierarchy.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Parameters
        ----------
        name : str
            Name of the hierarchy.
        boundaries : Sequence[LeveledBoundary]
            A list of leveled boundaries. They will be sorted by time upon initialization.
        """
        # Validate that all boundaries are LeveledBoundary instances
        for boundary in boundaries:
            if not isinstance(boundary, LeveledBoundary):
                raise TypeError("All boundaries must be LeveledBoundary instances")

        # Call parent constructor which handles sorting and TimeSpan initialization
        super().__init__(name=name, boundaries=boundaries)

    def __getitem__(self, key: int) -> LeveledBoundary:
        return self.boundaries[key]

    def to_multisegment(self) -> MultiSegment:
        """
        Convert the BoundaryHierarchy to a MultiSegment.

        The MultiSegment will have layers from coarsest (highest level) to finest (lowest level).
        Empty strings for all labels.
        """
        layers = []
        max_level = max(b.level for b in self.boundaries)
        for level in range(max_level, 0, -1):
            level_boundaries = [Boundary(b.time) for b in self.boundaries if b.level >= level]
            labels = [""] * (len(level_boundaries) - 1)
            layers.append(Segment(boundaries=level_boundaries, labels=labels, name=f"L{max_level - level + 1:02d}"))

        return MultiSegment(layers=layers, name=self.name)


# endregion
