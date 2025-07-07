"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jams
from matplotlib.axes import Axes

from bnl import viz

# region: Point-like Objects


@dataclass(frozen=True, order=True)
class Boundary:
    """
    A divider in the flow of time in seconds, quantized to 1e-5 seconds.
    """

    time: float

    def __post_init__(self):
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

    def __post_init__(self):
        if self.end.time <= self.start.time:
            raise ValueError("TimeSpan must have a non-zero, positive duration.")

    @property
    def duration(self) -> float:
        return self.end.time - self.start.time

    def __repr__(self) -> str:
        return f"TS({self.start}-{self.end}, {self.name})"

    def __str__(self) -> str:
        return self.name if self.name != "" else f"[{self.start.time:.2f}-{self.end.time:.2f}]"

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """
        Plots the time span on a set of axes.

        A wrapper around `bnl.viz.plot_timespan`.
        """
        return viz.plot_timespan(self, ax=ax, **kwargs)


class Segment(TimeSpan):
    """
    An ordered sequence of boundaries that partition a span into labeled sections.
    Represents one layer of annotation.
    """

    def __init__(self, boundaries: list[Boundary], labels: list[str], name: str = "Segment"):
        """
        Initializes the Segment.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Parameters
        ----------
        boundaries : list[Boundary]
            A list of at least two boundaries, sorted by time.
        labels : list[str]
            A list of labels for the sections. Must be `len(boundaries) - 1`.
        name : str, optional
            Name of the segment. Defaults to "Segment".
        """
        if not boundaries or len(boundaries) < 2:
            raise ValueError("A Segment requires at least two boundaries.")
        if len(labels) != len(boundaries) - 1:
            raise ValueError("Number of labels must be one less than the number of boundaries.")
        if boundaries != sorted(boundaries):
            raise ValueError("Boundaries must be sorted by time.")

        self.boundaries = boundaries
        self.labels = labels
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    @property
    def sections(self) -> list[TimeSpan]:
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
    def from_itvls(cls, itvls: list[list[float]], labels: list[str], name: str = "Segment") -> Segment:
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
        return viz.plot_segment(self, ax=ax, **kwargs)


class MultiSegment(TimeSpan):
    """
    The primary input object for analysis, containing multiple Segment layers.
    """

    def __init__(self, layers: list[Segment], name: str = "Hierarchical Segmentation"):
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
        if not layers:
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
        for i, layer in enumerate(json_data):
            itvls, labels = layer
            layers.append(Segment.from_itvls(itvls, labels, name=f"L{i:02d}"))
        return cls(layers=layers, name=name)

    def to_contour(self) -> BoundaryContour:
        pass

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """
        Plots the MultiSegment on an axes.

        A wrapper around `bnl.viz.plot_multisegment`.
        """
        return viz.plot_multisegment(self, ax=ax, **kwargs)

    @staticmethod
    def align_layers(layers: list[Segment]) -> list[Segment]:
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

    def __init__(self, name: str, boundaries: list[RatedBoundary]):
        """
        Initializes the BoundaryContour.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Parameters
        ----------
        name : str
            Name of the contour.
        boundaries : list[RatedBoundary]
            A list of rated boundaries. They will be sorted by time upon initialization.
        """
        self.boundaries = sorted(boundaries)
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    def __len__(self) -> int:
        return len(self.boundaries)

    def __getitem__(self, key: int) -> RatedBoundary:
        return self.boundaries[key]

    def to_levels(self) -> BoundaryHierarchy:
        pass


class BoundaryHierarchy(TimeSpan):
    """
    The structural output of the monotonic casting process.
    """

    def __init__(self, name: str, boundaries: list[LeveledBoundary]):
        """
        Initializes the BoundaryHierarchy.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Parameters
        ----------
        name : str
            Name of the hierarchy.
        boundaries : list[LeveledBoundary]
            A list of leveled boundaries. They will be sorted by time upon initialization.
        """
        self.boundaries = sorted(boundaries)
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    def __len__(self) -> int:
        return len(self.boundaries)

    def __getitem__(self, key: int) -> LeveledBoundary:
        return self.boundaries[key]


# endregion
