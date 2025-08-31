"""Core data structures for monotonic boundary casting."""

from __future__ import annotations

from collections import Counter
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
from functools import cached_property
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


# region: TimeSpan and Segment


@dataclass(frozen=True)
class TimeSpan:
    """An abstract base class for objects that represent a span of time.

    This class defines the interface for all time-spanned objects, ensuring
    they have `start`, `end`, `duration`, and `name` properties.
    """

    start: Boundary
    end: Boundary
    name: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        self._validate_timespan()
        if self.name is None:
            object.__setattr__(self, "name", self._interval_str())

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end.time - self.start.time

    def _validate_timespan(self):
        """Helper method to validate that the timespan is valid."""
        if self.end.time <= self.start.time:
            raise ValueError("TimeSpan must have a non-zero, positive duration.")

    def _interval_str(self) -> str:
        """Helper method to generate a default name from the boundaries."""
        return f"[{self.start.time:.2f}-{self.end.time:.2f}]"

    def __repr__(self) -> str:
        return f"TS({self._interval_str()}, {self.name})"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Segment(TimeSpan):
    """An ordered sequence of boundaries that partition a span into labeled sections.

    Represents one layer of annotation. While it inherits from `TimeSpan`,
    its `start` and `end` attributes are automatically derived from the provided
    `boundaries`.
    """

    bs: Sequence[Boundary]
    raw_labs: Sequence[str | None] = field(default_factory=list)
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self) -> None:
        """Validates the core assumptions of the Segment."""
        if not self.bs or len(self.bs) < 2:
            raise ValueError("A Segment requires at least two boundaries.")
        if len(self.raw_labs) == 0:
            object.__setattr__(self, "raw_labs", [None] * (len(self.bs) - 1))
        if len(self.raw_labs) != len(self.bs) - 1:
            raise ValueError(
                f"Number of labels ({len(self.raw_labs)}) must be one less than "
                f"the number of boundaries ({len(self.bs)})"
            )
        if any(self.bs[i] > self.bs[i + 1] for i in range(len(self.bs) - 1)):
            raise ValueError(f"Boundaries must be sorted. {self.bs}")

        # Use object.__setattr__ to assign to the init=False fields.
        object.__setattr__(self, "start", self.bs[0])
        object.__setattr__(self, "end", self.bs[-1])
        super().__post_init__()

    @cached_property
    def sections(self) -> Sequence[TimeSpan]:
        """A list of all the labeled time spans that compose the segment."""
        return [
            TimeSpan(name=label, start=Boundary(itvl[0]), end=Boundary(itvl[1]))
            for itvl, label in zip(self.itvls, self.raw_labs)
        ]

    @cached_property
    def labels(self) -> Sequence[str]:
        return [s.name for s in self.sections]

    @cached_property
    def itvls(self) -> np.ndarray:
        itvls = [[b.time, e.time] for b, e in zip(self.bs[:-1], self.bs[1:])]
        return np.array(itvls)

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
        return f"S({self._interval_str()}, {self.name})"

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_jams(cls, segment_annotation: jams.Annotation, name: str | None = None) -> Segment:
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
        name: str | None = None,
    ) -> Segment:
        """Data Ingestion from `mir_eval` format of boundaries and labels."""
        # assume intervals have no overlap or gaps
        bs = [Boundary(itvl[0]) for itvl in itvls]
        # tag on the end time of the last interval
        bs.append(Boundary(itvls[-1][1]))
        return cls(bs, labels, name=name)

    @classmethod
    def from_bs(
        cls,
        bs: Sequence[Boundary | Number],
        labels: Sequence[str] | None = None,
        name: str | None = None,
    ) -> Segment:
        """Creates a Segment from a sequence of boundaries and labels."""
        bs = [Boundary(b) if isinstance(b, Number) else b for b in bs]
        if labels is None:
            labels = []
        return cls(bs, labels, name=name)

    def plot(
        self,
        colorscale: str | list[str] = "D3",
        hatch: bool = True,
    ) -> go.Figure:
        """Plots the segment on a plotly figure by warpping it in a MultiSegment."""
        ms = MultiSegment(raw_layers=[self], name=str(self))
        fig = ms.plot(colorscale=colorscale, hatch=hatch)
        fig.update_layout(yaxis_visible=False)
        return fig

    def scrub_labels(self, replace_with: str | None = "") -> Segment:
        """Scrubs the labels of the Segment by replacing them with empty strings."""
        return replace(self, raw_labs=[replace_with] * len(self.raw_labs))

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
@dataclass(frozen=True)
class MultiSegment(TimeSpan):
    """The primary input object for analysis, containing multiple Segment layers."""

    raw_layers: Sequence[Segment] = field(default_factory=list)
    """A sequence of `Segment` objects representing different layers of annotation."""
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self):
        """Validates the core assumptions of the MultiSegment."""
        if not self.raw_layers:
            raise ValueError("MultiSegment must contain at least one Segment layer.")

        # Calculate the unified span and set the start/end boundaries.
        unified_span = self.find_span(self.raw_layers, mode="union")
        object.__setattr__(self, "start", unified_span.start)
        object.__setattr__(self, "end", unified_span.end)
        super().__post_init__()

    @cached_property
    def layers(self) -> Sequence[Segment]:
        """Returns the layers aligned to a unified time span."""
        # put all raw_layers on a unified time span
        unified_span = TimeSpan(self.start, self.end)
        aligned_layers = [layer.align(unified_span) for layer in self.raw_layers]

        # make sure all layer's name are distinct, if not, add suffix based on occurrence count.
        seen_names_count = Counter()
        processed_layers = []
        for layer in aligned_layers:
            count = seen_names_count[layer.name]
            seen_names_count[layer.name] += 1
            if count:
                layer = replace(layer, name=f"{layer.name}_{count}")
            processed_layers.append(layer)

        return processed_layers

    def __len__(self) -> int:
        return len(self.raw_layers)

    def __getitem__(self, key: int) -> Segment:
        return self.layers[key]

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.layers)

    @property
    def itvls(self) -> Sequence[np.ndarray]:
        """Returns a list of all the intervals for each layer in the MultiSegment."""
        return [layer.itvls for layer in self]

    @property
    def labels(self) -> Sequence[Sequence[str]]:
        """Returns a list of all the labels for each layer in the MultiSegment."""
        return [layer.labels for layer in self]

    @classmethod
    def from_json(cls, json_data: list, name: str | None = None) -> MultiSegment:
        """Data Ingestion from adobe json format.

        Args:
            json_data (list): A list of layers, where each layer is a tuple of
                (intervals, labels). `intervals` is a list of [start, end] times,
                and `labels` is a list of strings.
            name (str, optional): Name for the created MultiSegment.
        """
        layers = []
        for i, layer in enumerate(json_data, start=1):
            itvls, labels = layer
            layers.append(Segment.from_itvls(itvls, labels, name=f"L{i:02d}"))
        return cls(raw_layers=layers, name=name)

    @classmethod
    def from_itvls(
        cls, itvls: Sequence[Sequence[float]], labels: Sequence[str], name: str | None = None
    ) -> MultiSegment:
        layers = []
        for i in range(len(itvls)):
            layers.append(Segment.from_itvls(itvls[i], labels[i], name=f"L{i + 1:02d}"))
        return cls(raw_layers=layers, name=name)

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
        contour_strategy = strategy_class(**kwargs)
        return contour_strategy(self)

    def scrub_labels(self, replace_with: str | None = "") -> MultiSegment:
        """Scrubs the labels of the MultiSegment by replacing them with empty strings."""
        return MultiSegment(
            raw_layers=[layer.scrub_labels(replace_with) for layer in self], name=self.name
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
        return MultiSegment(raw_layers=[layer.align(span) for layer in self], name=self.name)

    def prune_layers(self, relabel: bool = True) -> MultiSegment:
        """Prunes identical layers from the MultiSegment.

        This also gets rid of layers with no inner boundaries.
        """
        pruned_layers = []
        for layer in self:
            if len(layer) <= 1:
                continue  # skip layers with no inner boundaries

            # The first valid layer is always added.
            if not pruned_layers:
                pruned_layers.append(layer)
                continue

            # Subsequent layers are added only if they differ from the previous one.
            same_boundaries = np.array_equal(layer.bs, pruned_layers[-1].bs)
            same_labeling = np.array_equal(layer.lam, pruned_layers[-1].lam)
            if not (same_boundaries and same_labeling):
                pruned_layers.append(layer)

        final_layers = pruned_layers
        if relabel:
            final_layers = [
                replace(layer, name=f"L{i:02d}") for i, layer in enumerate(pruned_layers, start=1)
            ]

        return replace(self, raw_layers=final_layers)

    def squeeze_layers(self, times: int = 1, relabel: bool = True) -> MultiSegment:
        """Remove the least informative layer from the MultiSegment according to vmeasure.

        Returns a new MultiSegment with the most redundant layer removed.
        """
        from mir_eval.segment import vmeasure

        if times <= 0 or len(self) <= 1:
            return self
        elif times == 1:
            # base case:
            # get rid of the level that adds the least information
            # look at vmeasure between all consecutive levels,
            # get the one with the highest vmeasure with the next level
            v_f1 = [
                vmeasure(lv1.itvls, lv1.labels, lv2.itvls, lv2.labels)[2]
                for lv1, lv2 in zip(self, self[1:])
            ]
            idx_to_pop = np.argmax(v_f1)
            new_layers = [layer for i, layer in enumerate(self) if i != idx_to_pop]
            if relabel:
                new_layers = [
                    replace(layer, name=f"L{i:02d}") for i, layer in enumerate(new_layers, start=1)
                ]
            return replace(self, raw_layers=new_layers)
        else:
            # Recurse
            return self.squeeze_layers(times - 1, relabel=relabel).squeeze_layers(
                times=1, relabel=relabel
            )

    def meet(self) -> tuple[np.ndarray, np.ndarray]:
        import frameless_eval as fle
        import mir_eval

        grid, labels, _ = fle.utils.make_common_itvls(self.itvls, self.labels, [], [])
        boundaries = mir_eval.util.intervals_to_boundaries(grid)
        return boundaries, fle.utils.meet(labels)


# endregion: MultiSegment


# region: Monotonic Boundary
@dataclass(frozen=True)
class BoundaryContour(TimeSpan):
    """
    An intermediate, purely structural representation of boundary salience over time.
    """

    bs: Sequence[RatedBoundary]
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self):
        if not self.bs or len(self.bs) < 2:
            raise ValueError("A BoundaryContour requires at least two boundaries.")
        if any(self.bs[i].time > self.bs[i + 1].time for i in range(len(self.bs) - 1)):
            raise ValueError(f"Boundaries must be sorted. {self.bs}")

        # Use object.__setattr__ to assign to the init=False fields.
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
        clean_strategy = strategy_class(**kwargs)

        return clean_strategy(self)

    def level(self, strategy: str = "unique", **kwargs: Any) -> BoundaryHierarchy:
        """
        Converts the BoundaryContour to a BoundaryHierarchy by quantizing salience.
        """
        from . import ops

        if strategy not in ops.LevelStrategy._registry:
            raise ValueError(f"Unknown boundary level strategy: {strategy}")

        strategy_class = ops.LevelStrategy._registry[strategy]
        level_strategy = strategy_class(**kwargs)

        return level_strategy(self)


@dataclass(frozen=True)
class BoundaryHierarchy(BoundaryContour):
    """
    The structural output of the monotonic casting process.
    """

    bs: Sequence[LeveledBoundary]
    start: Boundary = field(init=False)
    end: Boundary = field(init=False)

    def __post_init__(self):
        for boundary in self.bs:
            if not isinstance(boundary, LeveledBoundary):
                raise TypeError("All boundaries must be LeveledBoundary instances")

        # Use object.__setattr__ to assign to the init=False fields.
        object.__setattr__(self, "start", self.bs[0])
        object.__setattr__(self, "end", self.bs[-1])
        super().__post_init__()

    def to_ms(self, name: str | None = None) -> MultiSegment:
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
            labels = [None] * (len(level_boundaries) - 1)
            layers.append(
                Segment(
                    bs=level_boundaries,
                    raw_labs=labels,
                    name=f"L{max_level - level + 1:02d}",
                )
            )

        return MultiSegment(layers, name=name or f"{self.name} Monotonic MS")


# endregion
