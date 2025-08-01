"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

from numbers import Number

__all__ = [
    "Boundary",
    "RatedBoundary",
    "LeveledBoundary",
    "TimeSpan",
    "Segment",
    "MultiSegment",
    "BoundaryContour",
    "BoundaryHierarchy",
]

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jams
import numpy as np
import plotly.graph_objects as go

# region: Boundary Objects


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
        return f"B({self.time:.1f})"


@dataclass(frozen=True, order=True)
class RatedBoundary(Boundary):
    """
    A boundary with a continuous measure of importance or salience.
    """

    salience: float

    def __repr__(self) -> str:
        return f"RB({self.time:.1f}, {self.salience:.2f})"


@dataclass(frozen=True, order=True, init=False)
class LeveledBoundary(RatedBoundary):
    """A boundary that exists in a monotonic hierarchy.

    This object represents a boundary that has been assigned a discrete
    hierarchical level. The `salience` attribute is automatically set
    to be equal to the `level`.
    """

    #: The discrete hierarchical level of the boundary.
    level: int

    def __init__(self, time: float, level: int):
        """
        Args:
            time (float): The time of the boundary in seconds.
            level (int): The discrete hierarchical level of the boundary.
                         Must be a positive integer.

        Raises:
            ValueError: If `level` is not a positive integer.
        """
        if not isinstance(level, int) or level <= 0:
            raise ValueError(
                f"`level` must be a positive integer, got {level}, with type {type(level)}."
            )

        # Manually set the attributes for this frozen instance.
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "salience", float(level))

        # Explicitly call the Boundary object's post-init for time validation.
        super().__post_init__()

    def __repr__(self) -> str:
        return f"LB({self.time:.1f}, {self.level})"


# endregion


# region: Simple Interval Objects


@dataclass
class TimeSpan:
    """
    Represents a generic time interval.

    Must have a non-zero, positive duration. Allows empty string for name as default.
    """

    start: Boundary
    end: Boundary
    # docstring: Name of the time span, defaults to `[start-end]` if None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.end.time <= self.start.time:
            raise ValueError("TimeSpan must have a non-zero, positive duration.")
        if self.name is None:
            self.name = f"[{self.start.time:.2f}-{self.end.time:.2f}]"

    @property
    def duration(self) -> float:
        return self.end.time - self.start.time

    def __repr__(self) -> str:
        return f"TS({self.start}-{self.end}, {self.name})"

    def __str__(self) -> str:
        return self.name


@dataclass
class Segment(TimeSpan):
    """An ordered sequence of boundaries that partition a span into labeled sections.

    Represents one layer of annotation. While it inherits from `TimeSpan`,
    its `start` and `end` attributes are automatically derived from the provided
    `boundaries`.
    """

    bs: Sequence[Boundary] = field(default_factory=list)
    labels: Sequence[str] = field(default_factory=list)
    name: str = "Segment"

    # Exclude parent fields from the __init__ signature, they are derived.
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self) -> None:
        """Validates and initializes the derived fields of the Segment."""
        if not self.bs or len(self.bs) < 2:
            raise ValueError("A Segment requires at least two boundaries.")
        if not self.labels:
            raise ValueError("Segment requires labels.")
        if len(self.labels) != len(self.bs) - 1:
            raise ValueError("Number of labels must be one less than the number of boundaries.")

        # Ensure boundaries is sorted.
        self.bs = sorted(self.bs)
        # Derive and set the parent's start and end fields.
        self.__setattr__("start", self.bs[0])
        self.__setattr__("end", self.bs[-1])
        # Call the parent's post-init to validate duration.
        super().__post_init__()

        # Create cache to avoid repeated computation.
        self._sections = None
        self._itvls = None

    @property
    def sections(self) -> Sequence[TimeSpan]:
        """A list of all the labeled time spans that compose the segment."""
        if self._sections is None:
            self._sections = [
                TimeSpan(start=Boundary(itvl[0]), end=Boundary(itvl[1]), name=label)
                for itvl, label in zip(self.itvls, self.labels)
            ]
        return self._sections

    @property
    def itvls(self) -> np.ndarray:
        if self._itvls is None:
            itvls = [[b.time, e.time] for b, e in zip(self.bs[:-1], self.bs[1:])]
            self._itvls = np.array(itvls)
        return self._itvls

    @property
    def lam(self) -> np.ndarray:
        """Label Agreement Matrix

        Returns:
            np.ndarray: The label agreement matrix.
        """
        return np.equal.outer(self.labels, self.labels)

    def __len__(self) -> int:
        return len(self.sections)

    def __getitem__(self, key: int) -> TimeSpan:
        return self.sections[key]

    def __iter__(self) -> Iterator[TimeSpan]:
        return iter(self.sections)

    def __repr__(self) -> str:
        return f"S({self.start}-{self.end}, {self.name})"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_jams(cls, segment_annotation: jams.Annotation, name: str = "Segment") -> Segment:
        """
        Data Ingestion from jams format.
        """
        itvls, labels = segment_annotation.to_interval_values()
        return cls.from_itvls(itvls, labels, name=name)

    @classmethod
    def from_itvls(
        cls,
        itvls: Sequence[Sequence[float]],
        labels: Sequence[str],
        name: str = "Segment",
    ) -> Segment:
        """Data Ingestion from `mir_eval` format of boundaries and labels."""
        # assume intervals have no overlap or gaps
        bs = [Boundary(itvl[0]) for itvl in itvls]
        # tag on the end time of the last interval
        bs.append(Boundary(itvls[-1][1]))
        return cls(bs=bs, labels=labels, name=name)

    @classmethod
    def from_bs(
        cls,
        bs: Sequence[Boundary | Number],
        labels: Sequence[str],
        name: str = "Segment",
    ) -> Segment:
        """Creates a Segment from a sequence of boundaries and labels."""
        bs = [Boundary(b) if isinstance(b, Number) else b for b in bs]
        return cls(bs=bs, labels=labels, name=name)

    def plot(
        self,
        colorscale: str | list[str] = "D3",
        hatch: bool = True,
    ) -> go.Figure:
        """Plots the segment on a plotly figure by warpping it in a MultiSegment."""
        ms = MultiSegment(layers=[self], name=str(self))
        fig = ms.plot(colorscale=colorscale, hatch=hatch)
        fig.update_layout(yaxis_visible=False)
        return fig

    def scrub_labels(self, replace_with: str | None = "") -> Segment:
        """Scrubs the labels of the Segment by replacing them with empty strings."""
        return replace(self, labels=[replace_with] * len(self.labels))

    def align(self, span: TimeSpan) -> Segment:
        """Align with a TimeSpan object."""
        if len(self.bs) == 2:
            return replace(self, bs=[span.start, span.end])

        inner_bs = self.bs[1:-1]
        if span.start.time >= inner_bs[0].time or span.end.time <= inner_bs[-1].time:
            raise ValueError(f"New span {span} does not contain the inner boundaries.")

        new_bs = [span.start] + list(inner_bs) + [span.end]
        return replace(self, bs=new_bs)


# endregion: Segment


# region: MultiSegment
@dataclass
class MultiSegment(TimeSpan):
    """The primary input object for analysis, containing multiple Segment layers."""

    layers: Sequence[Segment] = field(default_factory=list)
    name: str = "Hierarchical Segmentation"

    # Exclude parent fields from the __init__ signature, they are derived.
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self):
        """Validates and initializes the derived fields of the MultiSegment."""
        if not self.layers:
            raise ValueError("MultiSegment must contain at least one Segment layer.")

        # align all layers to the same time span
        unified_span = self.find_span(self.layers, mode="union")
        self.layers = [layer.align(unified_span) for layer in self.layers]

        # Derive and set the parent's start and end fields.
        object.__setattr__(self, "start", unified_span.start)
        object.__setattr__(self, "end", unified_span.end)

        # Call the parent's post-init to validate duration.
        super().__post_init__()

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, key: int) -> Segment:
        return self.layers[key]

    def __iter__(self) -> Iterator[Segment]:
        """Enable iteration over the layers."""
        return iter(self.layers)

    @property
    def itvls(self) -> Sequence[np.ndarray]:
        """Returns a list of all the intervals in the MultiSegment."""
        return [layer.itvls for layer in self.layers]

    @property
    def labels(self) -> Sequence[Sequence[str]]:
        """Returns a list of all the labels in the MultiSegment."""
        return [layer.labels for layer in self.layers]

    @classmethod
    def from_json(cls, json_data: list, name: str | None = None) -> MultiSegment:
        """Data Ingestion from adobe json format.

        Args:
            json_data (list): A list of layers, where each layer is a tuple of
                (intervals, labels). `intervals` is a list of [start, end] times,
                and `labels` is a list of strings.
            name (str, optional): Name for the created MultiSegment.
                Defaults to "JSON Annotation".
        """
        layers = []
        for i, layer in enumerate(json_data, start=1):
            itvls, labels = layer
            layers.append(Segment.from_itvls(itvls, labels, name=f"L{i:02d}"))
        return cls(layers=layers, name=name if name is not None else "JSON Annotation")

    def plot(self, colorscale: str | list[str] = "D3", hatch: bool = True) -> go.Figure:
        """Plots the MultiSegment on a Plotly figure.

        Args:
            colorscale (str | list[str], optional): Plotly colorscale to use. Can be a
                qualitative scale name (e.g., "Set3", "Pastel") or a list of colors.
            hatch (bool, optional): Whether to use hatch patterns for different
                labels. Defaults to True.
        """
        from . import viz

        return viz.plot_multisegment(ms=self, colorscale=colorscale, hatch=hatch)

    def contour(self, strategy: str = "depth", **kwargs: Any) -> BoundaryContour:
        """Calculates boundary salience and converts to a BoundaryContour."""
        from . import ops

        if strategy not in ops.SalienceStrategy._registry:
            raise ValueError(f"Unknown salience strategy: {strategy}")

        strategy_class = ops.SalienceStrategy._registry[strategy]
        self._contour_strategy = strategy_class(**kwargs)

        return self._contour_strategy(self)

    def scrub_labels(self) -> MultiSegment:
        """Scrubs the labels of the MultiSegment by replacing them with empty strings."""
        return MultiSegment(
            layers=[layer.scrub_labels() for layer in self.layers],
            name=self.name,
        )

    @staticmethod
    def find_span(
        layers: Sequence[Segment],
        mode: str = "common",
    ) -> TimeSpan:
        """Finds the span of a list of Segment layers.

        Args:
            mode (str, optional): The alignment mode. Can be "union" or "common".
                Defaults to "common".
        """
        if mode == "union":
            inc_start_time = min(layer.start.time for layer in layers)
            inc_end_time = max(layer.end.time for layer in layers)
        elif mode == "common":
            inc_start_time = max(layer.start.time for layer in layers)
            inc_end_time = min(layer.end.time for layer in layers)
        else:
            raise ValueError(f"Unknown alignment mode: {mode}. Must be 'union' or 'common'.")
        return TimeSpan(
            start=Boundary(inc_start_time),
            end=Boundary(inc_end_time),
            name=f"{mode} span",
        )

    def align(self, span: TimeSpan) -> MultiSegment:
        """Align with a TimeSpan object."""
        return MultiSegment(
            layers=[layer.align(span) for layer in self.layers],
            name=self.name,
        )

    def prune_layers(self, relabel: bool = True) -> MultiSegment:
        """Prunes identical layers from the MultiSegment.

        This also gets rid of layers with no inner boundaries.
        """
        pruned_layers = []
        for layer in self.layers:
            if len(layer) <= 1:
                continue  # skip layers with no inner boundaries

            # using replace to hack a copy
            layer = replace(layer, name=layer.name)
            # The first valid layer is always added.
            if not pruned_layers:
                pruned_layers.append(layer)
                continue

            # Subsequent layers are added only if they differ from the previous one.
            same_boundaries = np.array_equal(layer.bs, pruned_layers[-1].bs)
            same_labeling = np.array_equal(layer.lam, pruned_layers[-1].lam)
            if not (same_boundaries and same_labeling):
                pruned_layers.append(layer)

        if relabel:
            for i, layer in enumerate(pruned_layers, start=1):
                layer.name = f"L{i:02d}"

        return replace(self, layers=pruned_layers)


# endregion: MultiSegment


# region: Monotonic Boundary
@dataclass
class BoundaryContour(TimeSpan):
    """
    An intermediate, purely structural representation of boundary salience over time.
    """

    bs: Sequence[RatedBoundary] = field(default_factory=list)
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)
    name: str = "BoundaryContour"

    def __post_init__(self):
        if not self.bs or len(self.bs) < 2:
            raise ValueError("A BoundaryContour requires at least two boundaries.")
        self.bs = sorted(self.bs)
        object.__setattr__(self, "start", self.bs[0])
        object.__setattr__(self, "end", self.bs[-1])
        super().__post_init__()

    def __len__(self) -> int:
        return len(self.bs) - 2

    def __getitem__(self, key: int) -> RatedBoundary:
        return self.bs[1:-1][key]

    def __iter__(self) -> Iterator[RatedBoundary]:
        return iter(self.bs[1:-1])

    def plot(self, **kwargs: Any) -> go.Figure:
        """Plots the BoundaryContour on a Plotly figure.

        Args:
            fig: Optional Plotly Figure to add to.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            A Plotly Figure object with the boundary contour visualization.
        """
        from . import viz

        return viz.plot_boundary_contour(self, **kwargs)

    def clean(self, strategy: str = "absorb", **kwargs: Any) -> BoundaryContour:
        """Cleans up the boundary contour using a specified strategy.

        This is a convenience wrapper around `bnl.ops.clean_boundaries`.

        Args:
            strategy (str): The cleaning strategy to use. See `bnl.ops.clean_boundaries`
                for details. Defaults to 'absorb'.
            **kwargs: Additional keyword arguments to pass to the strategy (e.g., `window`).

        Returns:
            BoundaryContour: A new, cleaned BoundaryContour.
        """
        from . import ops

        if strategy not in ops.CleanStrategy._registry:
            raise ValueError(f"Unknown boundary cleaning strategy: {strategy}")

        # Retrieve the class from the registry and instantiate it with the provided arguments.
        strategy_class = ops.CleanStrategy._registry[strategy]
        self._clean_strategy = strategy_class(**kwargs)

        return self._clean_strategy(self)

    def level(self, strategy: str = "unique", **kwargs: Any) -> BoundaryHierarchy:
        """
        Converts the BoundaryContour to a BoundaryHierarchy by quantizing salience.
        """
        from . import ops

        if strategy not in ops.LevelStrategy._registry:
            raise ValueError(f"Unknown boundary level strategy: {strategy}")

        strategy_class = ops.LevelStrategy._registry[strategy]
        self._level_strategy = strategy_class(**kwargs)

        return self._level_strategy(self)


@dataclass
class BoundaryHierarchy(BoundaryContour):
    """
    The structural output of the monotonic casting process.
    """

    bs: Sequence[LeveledBoundary] = field(default_factory=list)
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)
    name: str = "BoundaryHierarchy"

    def __post_init__(self):
        for boundary in self.bs:
            if not isinstance(boundary, LeveledBoundary):
                raise TypeError("All boundaries must be LeveledBoundary instances")
        # Call parent's post-init after type validation
        super().__post_init__()

    def to_ms(self) -> MultiSegment:
        """Convert the BoundaryHierarchy to a MultiSegment.

        The MultiSegment will have layers from coarsest (highest level) to
        finest (lowest level), with empty strings for all labels.

        Returns:
            MultiSegment: The resulting MultiSegment object.
        """
        layers = []
        max_level = max(b.level for b in self.bs)
        for level in range(max_level, 0, -1):
            level_boundaries = [Boundary(b.time) for b in self.bs if b.level >= level]
            labels = [""] * (len(level_boundaries) - 1)
            layers.append(
                Segment(
                    bs=level_boundaries,
                    labels=labels,
                    name=f"L{max_level - level + 1:02d}",
                )
            )

        return MultiSegment(layers=layers, name=f"{self.name} Monotonic MS")


# endregion
