"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

from dataclasses import dataclass

# I. Point-like Objects
# ---------------------
# These are the fundamental, label-less structural markers on the timeline.


@dataclass(frozen=True, order=True)
class Boundary:
    """A raw, unannotated marker on a timeline."""

    time: float

    def __post_init__(self):
        """Validates and quantizes the time attribute."""
        if self.time < 0:
            raise ValueError("Time cannot be negative.")

        # Round to 5 decimal places to avoid floating point precision issues.
        # Use object.__setattr__ because the dataclass is frozen.
        rounded_time = round(self.time, 5)
        object.__setattr__(self, "time", rounded_time)


@dataclass(frozen=True, order=True)
class RatedBoundary(Boundary):
    """A boundary with a continuous measure of importance or salience."""

    salience: float


@dataclass(frozen=True, order=True, init=False)
class LeveledBoundary(RatedBoundary):
    """
    A definitive structural node within a monotonic hierarchy.

    The `level` must be a positive integer, and the `salience` attribute
    is automatically set to be equal to the `level`.
    """

    level: int

    def __init__(self, time: float, level: int):
        """Initializes a LeveledBoundary, deriving salience from level."""
        if not isinstance(level, int) or level <= 0:
            raise ValueError("`level` must be a positive integer.")

        # Manually set the attributes for this frozen instance.
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "salience", float(level))

        # Explicitly call the Boundary object's post-init for time validation.
        super().__post_init__()


# II. Span-like Objects (Containers)
# ----------------------------------
# These objects represent time intervals and contain the point-like objects
# and their associated labels.


class TimeSpan:
    """
    Represents a generic time interval.

    Must have a non-zero, positive duration. If a name is not provided,
    it defaults to a string representation of the span (e.g., "[0.00-15.32]").
    """

    def __init__(self, start: Boundary, end: Boundary, name: str | None = None):
        if not isinstance(start, Boundary) or not isinstance(end, Boundary):
            raise TypeError("`start` and `end` must be Boundary objects.")
        if end.time <= start.time:
            raise ValueError("TimeSpan must have a non-zero, positive duration.")

        self.start = start
        self.end = end
        self.name = name if name is not None else f"[{start.time:.2f}-{end.time:.2f}]"

    @property
    def duration(self) -> float:
        """The duration of the time span."""
        return self.end.time - self.start.time

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start={self.start}, end={self.end}, name="{self.name}")'

    def plot(self):
        """The fundamental drawing method."""
        pass


class Segment(TimeSpan):
    """
    An ordered sequence of boundaries that partition a span into labeled sections.
    Represents one layer of annotation.
    """

    def __init__(self, name: str, boundaries: list[Boundary], labels: list[str]):
        if not boundaries or len(boundaries) < 2:
            raise ValueError("A Segment requires at least two boundaries.")
        if len(labels) != len(boundaries) - 1:
            raise ValueError("Number of labels must be one less than the number of boundaries.")

        self.boundaries = sorted(boundaries)
        self.labels = labels
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    @property
    def sections(self) -> list[TimeSpan]:
        """A list of all the labeled time spans that compose the segment."""
        return [
            TimeSpan(start=self.boundaries[i], end=self.boundaries[i + 1], name=self.labels[i])
            for i in range(len(self.labels))
        ]

    def plot(self):
        """Composes a plot by calling .plot() on each of its internal TimeSpan sections."""
        pass


class MultiSegment(TimeSpan):
    """The primary input object for analysis, containing multiple Segment layers."""

    def __init__(self, layers: list[Segment]):
        if not layers:
            raise ValueError("MultiSegment must contain at least one Segment layer.")

        self.layers = layers
        start_time = min(layer.start.time for layer in layers)
        end_time = max(layer.end.time for layer in layers)
        super().__init__(start=Boundary(start_time), end=Boundary(end_time), name="MultiSegment")

    @classmethod
    def from_jams(cls) -> MultiSegment:
        """Data Ingestion from JAMS file."""
        pass

    @classmethod
    def from_json(cls) -> MultiSegment:
        """Data Ingestion from JSON file."""
        pass

    def to_contour(self) -> BoundaryContour:
        """Aggregates boundaries into a single salience contour."""
        pass

    def plot(self):
        """The main, user-facing plotting method to visualize the input annotations."""
        pass


class BoundaryContour(TimeSpan):
    """An intermediate, purely structural representation of boundary salience over time."""

    def __init__(self, name: str, boundaries: list[RatedBoundary]):
        self.boundaries = sorted(boundaries)
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)

    def to_levels(self) -> BoundaryHierarchy:
        """Generates a set of leveled boundaries that adhere to monotonicity."""
        pass


class BoundaryHierarchy(TimeSpan):
    """The structural output of the monotonic casting process."""

    def __init__(self, name: str, boundaries: list[LeveledBoundary]):
        self.boundaries = sorted(boundaries)
        super().__init__(start=self.boundaries[0], end=self.boundaries[-1], name=name)
