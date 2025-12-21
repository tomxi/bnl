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
    "LabelAgreementMap",
    "SegmentAgreementProb",
    "SegmentAffinityMatrix",
]
import warnings
from abc import ABC
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import jams
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

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

    @classmethod
    def from_times(cls, start, end):
        return cls(Boundary(start), Boundary(end))


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
    def btimes(self) -> np.ndarray:
        return np.array([b.time for b in self.bs])

    @property
    def lam(self) -> LabelAgreementMap:
        """Label Agreement Matrix"""
        return LabelAgreementMap(mat=np.equal.outer(self.labels, self.labels), bs=self.btimes)

    @property
    def lam_pdf(self) -> LabelAgreementMap:
        """Label Agreement Matrix as probability density.
        np.sum(s.lam_pdf * lam_area_grid) == 1
        """
        lam_area = np.sum(self.lam.mat * self.lam.area_portion)
        return LabelAgreementMap(mat=self.lam.mat / lam_area, bs=self.btimes)

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
        return cls(bs=bs, raw_labs=labels, name=name)

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
        return cls(bs=bs, raw_labs=labels, name=name)

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

    def expand_labels(self) -> MultiSegment:
        """Keeping the same boundaries, expand and flatten labels to create a MultiSegment.
        According to the Hierarchy Expansion TISMIR 2021 paper by Katie and Brian.
        Using rules designed for the SALAMI dataset.
        """
        flat_labels = ["{}".format(_.replace("'", "")) for _ in self.labels]
        seg_counter = Counter()
        expanded_labels = []
        for label in flat_labels:
            expanded_labels.append(f"{label:s}_{seg_counter[label]:d}")
            seg_counter[label] += 1

        # create a flattened and a expanded layer
        flat_layer = Segment.from_itvls(self.itvls, flat_labels)
        expanded_layer = Segment.from_itvls(self.itvls, expanded_labels)
        return MultiSegment(raw_layers=[flat_layer, self, expanded_layer], name=self.name)

    def contour(self, normalize=True) -> BoundaryContour:
        btimes = self.btimes
        if len(btimes) <= 2:
            return BoundaryContour(name=self.name or "Boundary Contour", bs=self.bs)
        if normalize:
            weight = 1.0 / len(btimes[1:-1])
        else:
            weight = 1.0
        return BoundaryContour(
            name=self.name or "Boundary Contour",
            bs=[RatedBoundary(t, weight) for t in btimes],
        )


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

    @property
    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

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

    def to_json(self) -> list:
        json_data = []
        for layer in self:
            json_data.append([layer.itvls.tolist(), layer.labels])
        return json_data

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
            same_labeling = np.array_equal(layer.lam.mat, pruned_layers[-1].lam.mat)
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

    def meet(self) -> LabelAgreementMap:
        return self.lam(strategy="depth", mono=False)

    def lam(self, strategy: str = "depth", **kwargs: Any) -> LabelAgreementMap:
        from . import ops

        if strategy not in ops.LabelAgreementStrategy._registry:
            raise ValueError(f"Unknown label agreement strategy: {strategy}")

        strategy_class = ops.LabelAgreementStrategy._registry[strategy]
        lam_strategy = strategy_class(**kwargs)
        return lam_strategy(self)

    @staticmethod
    def _reindex_labels(ref_int, ref_lab, est_int, est_lab):
        # for each estimated label
        #    find the reference label that is maximally overlaps with
        score_map = defaultdict(lambda: 0)

        for r_int, r_lab in zip(ref_int, ref_lab):
            for e_int, e_lab in zip(est_int, est_lab):
                score_map[(e_lab, r_lab)] += max(
                    0, min(e_int[1], r_int[1]) - max(e_int[0], r_int[0])
                )

        r_taken = set()
        e_map = dict()

        hits = [(score_map[k], k) for k in score_map]
        hits = sorted(hits, reverse=True)

        while hits:
            cand_v, (e_lab, r_lab) = hits.pop(0)
            if r_lab in r_taken or e_lab in e_map:
                continue
            e_map[e_lab] = r_lab
            r_taken.add(r_lab)

        # Anything left over is unused
        unused = set(est_lab) - set(ref_lab)

        for e, u in zip(set(est_lab) - set(e_map.keys()), unused):
            e_map[e] = u

        return [e_map[e] for e in est_lab]

    def relabel(self):
        # relabel finer layers by finding max overlap with coarser layers
        new_labels = [self.labels[0]]
        for i in range(1, len(self.labels)):
            labs = self._reindex_labels(
                self.itvls[i - 1], new_labels[i - 1], self.itvls[i], self.labels[i]
            )
            new_labels.append(labs)

        return MultiSegment.from_itvls(self.itvls, new_labels, name=self.name)

    def expand_labels(self) -> MultiSegment:
        expanded_layers = []
        for layer in self:
            expanded_layers.extend(layer.expand_labels().layers)
        return MultiSegment(raw_layers=expanded_layers, name=self.name).prune_layers()

    def bpcs(self, bw=0.5, time_grid=None) -> pd.DataFrame:
        from . import ops

        if time_grid is None:
            time_grid = ops.build_time_grid(self, 0.1)
        return pd.concat(
            [layer.contour(normalize=True).bpc(bw=bw, time_grid=time_grid) for layer in self],
            axis=1,
        )

    def has_monotonic_bs(self) -> bool:
        coarser_layer = self[0]

        for finer_layer in self[1:]:
            # check if all btimes in coarser_layer are in finer_layer
            # if not, return false
            if not all(b in finer_layer.btimes for b in coarser_layer.btimes):
                return False
            # move down 1 layer
            coarser_layer = finer_layer
        return True

    def monocast(
        self,
        label_strat: str = "lam",
        w_b: pd.Series | None = None,
        w_l: pd.Series | None = None,
        **kwargs,
    ) -> MultiSegment:
        return (
            self.contour("prob", w=w_b)
            .clean("kde", bw=0.8)
            .level(strategy="mean_shift", log=True, bw=1 / 3)
            .to_ms(
                strategy=label_strat,
                ref_ms=self,
                w=w_l,
                **kwargs,
            )
        )


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

    def bpc(self, bw=0.5, time_grid=None, pmf=False) -> pd.Series:
        from sklearn.neighbors import KernelDensity

        from . import ops

        if time_grid is None:
            time_grid = ops.build_time_grid(self, 0.1)
        # get a Kernel Density Object and the log_density
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)

        effective_bs = self.bs[1:-1]
        if len(effective_bs) == 0:
            p = np.ones_like(time_grid) / self.duration
        else:
            times = np.array([b.time for b in effective_bs])
            saliences = np.array([b.salience for b in effective_bs])
            kde.fit(times.reshape(-1, 1), sample_weight=saliences)
            log_density = kde.score_samples(time_grid.reshape(-1, 1))
            p = np.exp(log_density)

        if pmf:
            p /= p.sum()
        return pd.Series(p, index=time_grid, name=self.name)


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

    def to_ms(self, strategy: str = "unique", **kwargs: Any) -> MultiSegment:
        """Convert the BoundaryHierarchy to a MultiSegment.

        The MultiSegment will have layers from coarsest (highest level) to
        finest (lowest level).

        strategy: "unique", "1layer", "lam"

        Returns:
            MultiSegment: The resulting MultiSegment object.
        """
        from . import ops

        if strategy not in ops.LabelingStrategy._registry:
            raise ValueError(f"Unknown labeling strategy: {strategy}")

        strategy_class = ops.LabelingStrategy._registry[strategy]
        return strategy_class(**kwargs)(self)


# endregion


# region: Label Agreement
@dataclass(frozen=True)
class AgreementMatrix(ABC):
    """Base class for agreement matrices."""

    mat: np.ndarray  # 2D array of agreement (shape = n_segs, n_segs)
    bs: np.ndarray | None = field(default=None)  # 1D array of boundary times (len = n_segs + 1)
    labels: np.ndarray | None = field(default=None)  # 1D array of labels (len = n_segs)

    def __post_init__(self):
        if self.bs is None:
            object.__setattr__(self, "bs", np.arange(self.mat.shape[0] + 1))
        if self.labels is None:
            object.__setattr__(self, "labels", np.arange(self.mat.shape[0]))

    def plot(self, ax=None, colorbar=True, **kwargs):
        from . import viz

        ax = viz.agreement_mat_mpl(self, ax=ax, **kwargs)
        if colorbar:
            ax.figure.colorbar(ax.collections[0], ax=ax)
        return ax

    @cached_property
    def area_portion(self):
        track_dur = self.bs[-1] - self.bs[0]
        seg_portion = np.diff(self.bs) / track_dur
        return np.outer(seg_portion, seg_portion)

    def sample(self, positions: np.ndarray) -> np.ndarray:
        """Sample from the agreement matrix.
        positions: (n_samples, 2) array of positions in bs
        returns: (n_samples, ) array of sampled values
        """
        # basic input gaurding
        if positions.shape[1] != 2:
            raise ValueError("positions must be of shape (n_samples, 2)")
        if np.any(positions < self.bs[0]) or np.any(positions > self.bs[-1]):
            raise ValueError("positions must be within the bounds of bs")
        # positions are in bs, convert to indices
        x_indices = np.searchsorted(self.bs, positions[:, 0], side="right") - 1
        y_indices = np.searchsorted(self.bs, positions[:, 1], side="right") - 1
        # if its the very end, use the last index
        x_indices[x_indices == len(self.bs) - 1] = len(self.bs) - 2
        y_indices[y_indices == len(self.bs) - 1] = len(self.bs) - 2
        return self.mat[x_indices, y_indices]


@dataclass(frozen=True)
class LabelAgreementMap(AgreementMatrix):
    """Label Agreement Map for segments. entries are PDFs"""

    def decode(
        self,
        bh: BoundaryHierarchy,
        aff_mode: str = "area",
        starting_k: int = 2,
        min_k_inc: int = 1,  # minimal k increment across layers
    ) -> MultiSegment:
        """Decode labels according to boundaries set out in bh
        Uses spectral clustering with k distinct labels.
        k is choosen by the Eigengap heuristic.
        k monotonically increase across layers, with a min_k_inc option.
        """
        new_layers = []
        current_k = starting_k
        for layer in bh.to_ms():
            aff = self.to_sap(layer.btimes).to_aff(aff_mode)
            current_k = aff.pick_k(min_k=current_k)
            lab = aff.scluster(k=current_k)
            # print(f"current_k: {current_k}")
            new_layers.append(Segment(bs=layer.bs, raw_labs=lab))
            current_k += min_k_inc
        return MultiSegment(raw_layers=new_layers, name=bh.name).relabel()

    def _upsample(self, finer_bs: np.ndarray) -> LabelAgreementMap:
        """Upsample the LabelAgreementMap to a new set of boundaries.
        finer_bs must be a superset of self.bs
        """
        if not np.all(np.isin(self.bs, finer_bs)):
            raise ValueError("finer_bs must be a superset of self.bs")
        if self.bs[0] != finer_bs[0] or self.bs[-1] != finer_bs[-1]:
            raise ValueError("finer_bs must have the same start and end as self.bs")

        # Find which original segment each new segment (start point) belongs to
        # segment_indices will be of length len(finer_bs) - 1
        segment_indices = np.searchsorted(self.bs, finer_bs[:-1], side="right") - 1

        # Use the mapping to create the new lam
        # finer_lam[i, j] = self.lam[segment_indices[i], segment_indices[j]]
        finer_lam = self.mat[segment_indices[:, None], segment_indices]

        return LabelAgreementMap(bs=finer_bs, mat=finer_lam)

    def _integrate(self, coarser_bs: np.ndarray) -> SegmentAgreementProb:
        """Integrate the LabelAgreementMap to a coarser set of boundaries."""
        from scipy.ndimage import sum_labels

        if not np.all(np.isin(coarser_bs, self.bs)):
            raise ValueError("coarser_bs must be a subset of self.bs")
        if self.bs[0] != coarser_bs[0] or self.bs[-1] != coarser_bs[-1]:
            raise ValueError("coarser_bs must have the same start and end as self.bs")

        # For each fine segment in self.bs, find which coarse segment it belongs to.
        segment_mapping = np.searchsorted(coarser_bs, self.bs[:-1], side="right") - 1

        # Create a 2D label matrix. Each element's value is a unique ID
        # for the coarse grid cell it belongs to (from 0 to n_coarse_segs**2 - 1).
        n_coarse_segs = len(coarser_bs) - 1
        label_matrix = (segment_mapping[:, None] * n_coarse_segs) + segment_mapping

        # lam is probability density, to integrate, we need to multiply each element
        # by the area_portion of the cell it belongs to.
        # Use sum_labels to calculate the sum of `self.mat` for each coarse cell.
        sums = sum_labels(
            self.mat * self.area_portion, labels=label_matrix, index=np.arange(n_coarse_segs**2)
        )
        # Reshape the 1D array of sums into the final 2D matrix
        new_matrix = sums.reshape((n_coarse_segs, n_coarse_segs))

        return SegmentAgreementProb(bs=coarser_bs, mat=new_matrix)

    def to_sap(self, bs=None) -> SegmentAgreementProb:
        """Integrate the LabelAgreementMap according to a set of boundaries.
        Effectively turning probability density into a PMF.
        This is useful for converting a LabelAgreementMap to a SegmentAgreementProb.
        """
        # respect the start and end of lam's bs
        track_boundary = (self.bs[0], self.bs[-1])
        if bs is None:
            bs = self.bs
        bs = np.concatenate([track_boundary, bs])
        bs = np.unique(np.clip(bs, a_min=track_boundary[0], a_max=track_boundary[-1]))

        # create common grid
        common_bs = np.unique(np.concatenate([self.bs, bs]))
        # upsample lam to common_bs, then integrate with respect to bs
        return self._upsample(common_bs)._integrate(bs)


@dataclass(frozen=True)
class SegmentAgreementProb(AgreementMatrix):
    """Label Agreement Matrix for segments. entries are PMFs"""

    def to_aff(self, normalize="area", self_link=True) -> SegmentAffinityMatrix:
        if not self_link:
            np.fill_diagonal(self.mat, 0)
        if normalize == "row":
            # Row-normalize the SAP matrix to get the transition matrix P
            row_sums = self.mat.sum(axis=1, keepdims=True)
            tran_mat = self.mat / (row_sums + 1e-9)  # Add epsilon for stability
            aff_mat = (tran_mat + tran_mat.T) / 2
        elif normalize == "area":
            aff_mat = self.mat / self.area_portion
        elif normalize == "cosine":
            aff_mat = cosine_similarity(self.mat)
        elif normalize == "area+cosine":
            aff_mat = cosine_similarity(self.mat / self.area_portion)
        else:
            raise ValueError(f"Unknown normalization mode: {normalize}")
        if not self_link:
            np.fill_diagonal(aff_mat, 0)
        return SegmentAffinityMatrix(aff_mat, self.bs)


@dataclass(frozen=True)
class SegmentAffinityMatrix(AgreementMatrix):
    """Label Agreement Matrix for segments. entries are affinities"""

    def laplacian(self, norm: str = "sym") -> np.ndarray:
        """
        Computes the graph Laplacian.
        This implementation correctly uses the full affinity matrix for degree calculation,
        as opposed to scipy's which ignores the main diagonal.
        """
        degrees = np.sum(self.mat, axis=1)
        unnormalized_laplacian = np.diag(degrees) - self.mat

        # Handle zero-degree nodes to avoid division by zero.
        # This is a robust way to prevent NaNs and Infs suggested by Gemini.
        with np.errstate(divide="ignore", invalid="ignore"):
            if norm == "rw":
                # Efficiently compute the Random Walk Laplacian: D^-1 * L
                inv_degrees = 1.0 / degrees
                inv_degrees[degrees == 0] = 0
                return np.diag(inv_degrees) @ unnormalized_laplacian

            elif norm == "sym":
                # Efficiently compute the Symmetric Normalized Laplacian: D^-1/2 * L * D^-1/2
                sqrt_inv_degrees = 1.0 / np.sqrt(degrees)
                sqrt_inv_degrees[degrees == 0] = 0
                d_sqrt_inv = np.diag(sqrt_inv_degrees)
                return d_sqrt_inv @ unnormalized_laplacian @ d_sqrt_inv

            else:
                raise NotImplementedError(
                    f"bad laplacian normalization mode: {norm}. has to be rw or sym"
                )

    def spec_decomp(self, lap_norm="sym") -> tuple[np.ndarray, np.ndarray]:
        from scipy.linalg import eig, eigh

        if lap_norm == "sym":
            evals, evecs = eigh(self.laplacian(lap_norm))
        elif lap_norm == "rw":
            # L_rw is generally not symmetric, use eig.
            evals, evecs = eig(self.laplacian(lap_norm))

            # Cast to real, and warn if imaginary parts are non-negligible.
            if not np.allclose(evals.imag, 0):
                warnings.warn(
                    "Eigenvalues have significant imaginary part", UserWarning, stacklevel=2
                )
            evals = evals.real
            if not np.allclose(evecs.imag, 0):
                warnings.warn(
                    "Eigenvectors have significant imaginary part", UserWarning, stacklevel=2
                )
            evecs = evecs.real

            # Sort the eigenvalues and eigenvectors
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
        else:
            raise ValueError(f"Unknown laplacian normalization mode: {lap_norm}")
        return evals, evecs

    @cached_property
    def lap_evals(self) -> np.ndarray:
        # evals for sym and rm normalized laplacian are the same. Use sym for speed.
        return self.spec_decomp("sym")[0]

    @property
    def egaps(self) -> np.ndarray:
        return np.diff(self.lap_evals)

    def scluster(self, k: int, lap_norm="sym") -> np.ndarray:
        from sklearn.cluster import KMeans

        n_segs = self.mat.shape[0]
        if n_segs == 1:
            return ["0"]
        if k >= n_segs:
            return [str(i) for i in range(n_segs)]

        evals, evecs = self.spec_decomp(lap_norm)
        first_k_evecs = evecs[:, :k]
        if lap_norm == "sym":
            norm = np.sqrt(np.sum(first_k_evecs**2, axis=1, keepdims=True))
            seg_embedding = first_k_evecs / (norm + 1e-8)
        elif lap_norm == "rw":
            seg_embedding = first_k_evecs
        else:
            raise ValueError(f"Unknown laplacian normalization mode: {lap_norm}")

        km = KMeans(n_clusters=k, init="k-means++", n_init=10)
        seg_ids = km.fit_predict(seg_embedding)
        return [str(i) for i in seg_ids]

    def named_clusters(self, k: int, lap_norm="sym") -> dict[str, list]:
        seg_ids = self.scluster(k, lap_norm)
        assignment = defaultdict(list)
        for i, key in enumerate(seg_ids):
            assignment[key].append(self.labels[i])
        return assignment

    def pick_k(
        self, min_k: int = 1, fiedler_threshold: float = 0.15, noise_threshold: float = 0.25
    ) -> int:
        """
        Determines k using eigengap
        """
        n_segs = self.mat.shape[0]
        if min_k >= n_segs or n_segs <= 1:
            return n_segs

        # candidate_gaps = self.egaps[min_k - 1 :]
        k = np.argmax(self.egaps) + 1

        # check the Fiedler value to decide if
        # we should return singleton groups or 1 big blob with k = 1
        if k == 1 and self.lap_evals[1] < fiedler_threshold:
            return n_segs
        # if k is already big enough, return it
        elif k >= min_k:
            return k
        # The largest gap occurred for a k < min_k.
        # We must find the best alternative k >= min_k.
        else:
            alt_k = np.argmax(self.egaps[min_k - 1 :]) + min_k
            # check if the new biggest gap is just noise, i.e. in the last 25% of the egaps
            if self.egaps[alt_k - 1] < np.percentile(self.egaps, 100 * noise_threshold):
                return n_segs
            else:
                return alt_k


# endregion: Label Agreement
