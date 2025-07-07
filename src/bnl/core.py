"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

from dataclasses import dataclass

# region: Point-like Objects


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


# endregion

# region: Span-like Objects (Containers)


@dataclass
class TimeSpan:
    """
    Represents a generic time interval.

    Must have a non-zero, positive duration. If a name is not provided,
    it defaults to a string representation of the span (e.g., "[0.00-15.32]").
    """

    start: Boundary
    end: Boundary
    name: str = ""

    def __post_init__(self):
        if self.end.time <= self.start.time:
            raise ValueError("TimeSpan must have a non-zero, positive duration.")

    @property
    def duration(self) -> float:
        """The duration of the time span."""
        return self.end.time - self.start.time

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start={self.start.time:.2f}, end={self.end.time:.2f}, name="{self.name}")'

    def plot(self):
        """The fundamental drawing method."""
        pass


class Segment(TimeSpan):
    """
    An ordered sequence of boundaries that partition a span into labeled sections.
    Represents one layer of annotation.
    """

    def __init__(self, boundaries: list[Boundary], labels: list[str], name: str = "Segment"):
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

    def plot(self):
        """Composes a plot by calling .plot() on each of its internal TimeSpan sections."""
        pass


class MultiSegment(TimeSpan):
    """The primary input object for analysis, containing multiple Segment layers."""

    def __init__(self, layers: list[Segment], name: str = "MultiSegment"):
        if not layers:
            raise ValueError("MultiSegment must contain at least one Segment layer.")

        self.layers = layers

        # All layers must span the same time interval.
        # Use the first layer as the reference for comparison.
        first_layer = layers[0]
        expected_start, expected_end = first_layer.start, first_layer.end

        for layer in layers[1:]:
            if layer.start != expected_start:
                raise ValueError("All layers must have the same start time.")
            if layer.end != expected_end:
                raise ValueError("All layers must have the same end time.")

        super().__init__(start=expected_start, end=expected_end, name=name)

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


# endregion
