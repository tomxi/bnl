"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

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
from dataclasses import dataclass, field
from typing import Any

import jams
import numpy as np
import plotly.graph_objects as go

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
        assert self.name is not None
        return self.name


@dataclass
class Segment(TimeSpan):
    """An ordered sequence of boundaries that partition a span into labeled sections.

    Represents one layer of annotation. While it inherits from `TimeSpan`,
    its `start` and `end` attributes are automatically derived from the provided
    `boundaries`.
    """

    bs: Sequence[Boundary] | None = None
    labels: Sequence[str] | None = None
    name: str = "Segment"

    # Exclude parent fields from the __init__ signature, they are derived.
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self):
        """Validates and initializes the derived fields of the Segment."""
        if self.bs is None or len(self.bs) < 2:
            raise ValueError("A Segment requires at least two boundaries.")
        if self.labels is None:
            raise ValueError("Segment requires labels.")
        if len(self.labels) != len(self.bs) - 1:
            raise ValueError(
                "Number of labels must be one less than the number of boundaries."
            )

        # Ensure boundaries is a mutable list and sorted.
        self.bs = list(self.bs)
        if self.bs != sorted(self.bs):
            raise ValueError("Boundaries must be sorted by time.")

        # Derive and set the parent's start and end fields.
        object.__setattr__(self, 'start', self.bs[0])
        object.__setattr__(self, 'end', self.bs[-1])

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
            itvls = [
                [b.time, e.time]
                for b, e in zip(self.bs[:-1], self.bs[1:])
            ]
            self._itvls = np.array(itvls)
        return self._itvls

    @property
    def lam(self) -> np.ndarray:
        """ Label Agreement Matrix
        
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

    @classmethod
    def from_jams(
        cls, segment_annotation: jams.Annotation, name: str = "Segment"
    ) -> Segment:
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
        """
        Data Ingestion from `mir_eval` format of boundaries and labels.
        """
        # assume intervals have no overlap or gaps
        bs = [Boundary(itvl[0]) for itvl in itvls]
        # tag on the end time of the last interval
        bs.append(Boundary(itvls[-1][1]))
        return cls(bs=bs, labels=labels, name=name)

    @classmethod
    def from_bs(
        cls,
        bs: Sequence[Boundary],
        labels: Sequence[str],
        name: str = "Segment",
    ) -> "Segment":
        """Creates a Segment from a sequence of boundaries and labels.

        This is a convenience constructor to allow for positional arguments.
        """
        return cls(bs=bs, labels=labels, name=name)

    def plot(
        self,
        colorscale: str | list[str] = "D3",
        hatch: bool = True,
    ) -> go.Figure:
        """
        Plots the segment on a plotly figure by warpping it in a MultiSegment.
        """
        ms = MultiSegment(layers=[self], name=str(self))
        fig = ms.plot(colorscale=colorscale, hatch=hatch)
        fig.update_layout(yaxis_visible=False)
        return fig

    def scrub_labels(self) -> Segment:
        """
        Scrubs the labels of the Segment by replacing them with empty strings.
        """

        return Segment(
            bs=self.bs,
            labels=[""] * len(self.labels),
            name=self.name,
        )

    

class MultiSegment(TimeSpan):
    """
    The primary input object for analysis, containing multiple Segment layers.
    """

    layers: Sequence[Segment]

    def __init__(
        self, layers: Sequence[Segment], name: str = "Hierarchical Segmentation"
    ):
        """Initializes the MultiSegment.

        The `start` and `end` attributes are automatically derived from the first layer.

        Args:
            layers (Sequence[Segment]): A list of Segment layers.
            name (str, optional): Name of the object.
        """
        if not layers:
            raise ValueError("MultiSegment must contain at least one Segment layer.")
        self.layers = self.align_layers(layers)
        super().__init__(start=self.layers[0].start, end=self.layers[0].end, name=name)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, key: int) -> Segment:
        return self.layers[key]

    def __iter__(self) -> Iterator[Segment]:
        """Enable iteration over the layers."""
        return iter(self.layers)

    @property
    def itvls(self) -> Sequence[np.ndarray]:
        return [layer.itvls for layer in self.layers]

    @property
    def labels(self) -> Sequence[Sequence[str]]:
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

        Returns:
            MultiSegment: A new MultiSegment instance.
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

        Returns:
            A Plotly Figure object with the multi-segment visualization.
        """
        from . import viz

        return viz.plot_multisegment(ms=self, colorscale=colorscale, hatch=hatch)

    def contour(self, strategy: str = "depth") -> BoundaryContour:
        """Calculates boundary salience and converts to a BoundaryContour.

        This is a convenience wrapper around `bnl.ops.boundary_salience`.

        Args:
            strategy (str, optional): The salience calculation strategy.
                See `bnl.ops.boundary_salience` for details. Defaults to 'depth'.

        Returns:
            BoundaryContour: The resulting boundary structure.
        """
        from . import ops  # Local import to avoid circular dependency at runtime

        return ops.boundary_salience(self, strategy=strategy)

    def scrub_labels(self) -> MultiSegment:
        """
        Scrubs the labels of the MultiSegment by replacing them with empty strings.
        """
        return MultiSegment(
            layers=[layer.scrub_labels() for layer in self.layers],
            name=self.name,
        )

    def align_layers(
        self,
        layers: Sequence[Segment],
        mode: str = "common",
    ) -> Sequence[Segment]:
        """Adjusts a list of Segment layers to have the same start and end as self.

        If self has no start or end, it will be inferred from the layers.
        This is achieved by finding the earliest start time and latest end time
        among all layers, and then extending each layer to this common span by stretching.

        Args:
            layers (Sequence[Segment]): The list of Segment layers to align.
            mode (str, optional): The alignment mode. Can be "union" or "common".

        Returns:
            Sequence[Segment]: A new list of aligned Segment objects.
        """
        if not layers:
            return []

        if mode == "union":
            inc_start_time = min(layer.start.time for layer in layers)
            inc_end_time = max(layer.end.time for layer in layers)
        elif mode == "common":
            inc_start_time = max(layer.start.time for layer in layers)
            inc_end_time = min(layer.end.time for layer in layers)
        else:
            raise ValueError(
                f"Unknown alignment mode: {mode}. Must be 'union' or 'common'."
            )

        start = getattr(self, "start", Boundary(inc_start_time))
        end = getattr(self, "end", Boundary(inc_end_time))

        aligned_layers = []
        for layer in layers:
            new_bs = layer.bs.copy()
            new_bs[0] = start
            new_bs[-1] = end

            aligned_layers.append(
                Segment(bs=new_bs, labels=layer.labels, name=layer.name or "")
            )

        if isinstance(layers, MultiSegment):
            return MultiSegment(layers=aligned_layers, name=layers.name)
        else:
            return aligned_layers

    def prune_layers(self, relabel: bool = True) -> MultiSegment:
        """Prunes identical layers from the MultiSegment.

        This also gets rid of layers with no inner boundaries.
        """
        pruned_layers = []
        for layer in self.layers:
            if len(layer) <= 1:
                continue # skip layers with no inner boundaries

            # The first valid layer is always added.
            if not pruned_layers:
                pruned_layers.append(layer.copy())
                continue

            # Subsequent layers are added only if they differ from the previous one.
            same_boundaries = np.array_equal(layer.bs, pruned_layers[-1].bs)
            same_labeling = np.array_equal(layer.lam, pruned_layers[-1].lam)
            if not (same_boundaries and same_labeling):
                pruned_layers.append(layer.copy())

        if relabel:
            for i, layer in enumerate(pruned_layers, start=1):
                layer.name = f"L{i:02d}"

        return MultiSegment(layers=pruned_layers, name=self.name)

class BoundaryContour(TimeSpan):
    """
    An intermediate, purely structural representation of boundary salience over time.
    """

    start: RatedBoundary
    end: RatedBoundary

    def __init__(self, name: str, boundaries: Sequence[RatedBoundary]):
        """Initializes the BoundaryContour.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Args:
            name (str): Name of the contour.
            boundaries (Sequence[RatedBoundary]): A list of rated boundaries.
                They will be sorted by time upon initialization.
        """
        if len(boundaries) < 2:
            raise ValueError("At least 2 boundaries for a TimeSpan!")
        self.bs: Sequence[RatedBoundary] = sorted(boundaries)
        super().__init__(start=self.bs[0], end=self.bs[-1], name=name)

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

        return ops.clean_boundaries(self, strategy=strategy, **kwargs)

    def level(self) -> BoundaryHierarchy:
        """
        Converts the BoundaryContour to a BoundaryHierarchy by quantizing salience.
        """
        from . import ops

        return ops.level_by_distinct_salience(self)


class BoundaryHierarchy(BoundaryContour):
    """
    The structural output of the monotonic casting process.
    """

    boundaries: Sequence[LeveledBoundary]

    def __init__(self, name: str, boundaries: Sequence[LeveledBoundary]):
        """Initializes the BoundaryHierarchy.

        The `start` and `end` attributes are automatically derived from the `boundaries` list.

        Args:
            name (str): Name of the hierarchy.
            boundaries (Sequence[LeveledBoundary]): A list of leveled boundaries.
                They will be sorted by time upon initialization.
        """
        # Validate that all boundaries are LeveledBoundary instances
        for boundary in boundaries:
            if not isinstance(boundary, LeveledBoundary):
                raise TypeError("All boundaries must be LeveledBoundary instances")

        # Call parent constructor which handles sorting and TimeSpan initialization
        super().__init__(name=name, boundaries=boundaries)

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
            level_boundaries = [
                Boundary(b.time) for b in self.bs if b.level >= level
            ]
            labels = [""] * (len(level_boundaries) - 1)
            layers.append(
                Segment(
                    boundaries=level_boundaries,
                    labels=labels,
                    name=f"L{max_level - level + 1:02d}",
                )
            )

        return MultiSegment(layers=layers, name=self.name or "BoundaryHierarchy")


# endregion
