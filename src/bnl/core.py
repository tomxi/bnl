"""Core data structures and constructors."""

from dataclasses import dataclass, field
from typing import Any

import jams
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from mir_eval.util import boundaries_to_intervals

__all__ = ["TimeSpan", "Segmentation", "Hierarchy"]


def _validate_time_value(value: Any, name: str) -> float:
    """Convert and validate a time value."""
    if isinstance(value, list) and not value:
        value = 0.0
    try:
        result = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to float, got {value} (type {type(value)})") from e

    result = np.round(result, 4)
    if result < 0:
        raise ValueError(f"{name} ({result}) cannot be negative.")
    return result


def _check_segments_contiguous(segments: list["TimeSpan"], name: str | None) -> None:
    """Check if segments are contiguous and non-overlapping for structural segmentations."""
    if not segments:
        return

    # Skip contiguity check for event-like namespaces
    event_namespaces = ["beat", "chord", "onset", "note", "key", "tempo", "lyrics", "chroma"]
    if name and any(event_ns in name.lower() for event_ns in event_namespaces):
        return

    for i in range(len(segments) - 1):
        curr_end, next_start = segments[i].end, segments[i + 1].start

        if not np.isclose(curr_end, next_start, atol=1e-9):
            if curr_end > next_start:
                raise ValueError(f"Segments must be non-overlapping. Overlap between {i} and {i + 1}.")
            else:
                raise ValueError(f"Segments must be contiguous. Gap between {i} and {i + 1}.")


@dataclass
class TimeSpan:
    """A labeled time span with start and end times."""

    start: float = 0.0
    end: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        self.start = _validate_time_value(self.start, "Start time")
        self.end = _validate_time_value(self.end, "End time")

        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must be ≤ end time ({self.end}).")

        if self.name is not None:
            self.name = str(self.name)

    @property
    def duration(self) -> float:
        """The duration of the time span (end - start)."""
        return np.round(self.end - self.start, 4)

    def __str__(self) -> str:
        lab = f": {self.name}" if self.name is not None else ""
        return f"TimeSpan([{self.start:.2f}s-{self.end:.2f}s], {self.duration:.2f}s{lab})"

    def __repr__(self) -> str:
        return f"TimeSpan(start={self.start}, end={self.end}, name='{self.name}')"

    def plot(
        self,
        ax: Axes | None = None,
        text: bool = True,
        **style_map: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Plot the time span as a vertical bar."""
        if ax is None:
            _, ax = plt.subplots()

        # Convert color to facecolor, set default edgecolor
        if "color" in style_map:
            style_map["facecolor"] = style_map.pop("color")
        style_map.setdefault("edgecolor", "white")

        rect = ax.axvspan(self.start, self.end, **style_map)

        if text and self.name:
            span_ymax = style_map.get("ymax", 1.0)
            ann = ax.annotate(
                self.name,
                xy=(self.start, span_ymax),
                xycoords=ax.get_xaxis_transform(),
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                clip_on=True,
                bbox=dict(boxstyle="round", facecolor="white"),
            )
            ann.set_clip_path(rect)

        return ax.figure, ax


@dataclass
class Segmentation(TimeSpan):
    """A segmentation containing multiple time spans."""

    segments: list[TimeSpan] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("Segmentation must contain at least one segment.")

        self.segments = sorted(self.segments, key=lambda x: x.start)
        _check_segments_contiguous(self.segments, self.name)

        self.start = self.segments[0].start
        self.end = self.segments[-1].end
        super().__post_init__()  # Validate start/end values

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx]

    @property
    def labels(self) -> list[str | None]:
        """A list of labels from all segments."""
        return [seg.name for seg in self.segments]

    @property
    def intervals(self) -> np.ndarray:
        """Intervals as a NumPy array of (start, end) pairs."""
        return np.array([[seg.start, seg.end] for seg in self.segments])

    @property
    def boundaries(self) -> list[float]:
        """A sorted list of unique boundary times."""
        boundaries = {seg.start for seg in self.segments} | {seg.end for seg in self.segments}
        return sorted(boundaries)

    def __repr__(self) -> str:
        name_str = f"name='{self.name}', " if self.name else ""
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        name_str = f"name='{self.name}', " if self.name else ""
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

    def plot(self, ax: Axes | None = None, text: bool = True, **kwargs: Any) -> tuple[Figure | SubFigure, Axes]:
        """Plot the segmentation using bnl.viz.plot_segment."""
        from .viz import plot_segment

        # Extract viz-specific parameters
        title = kwargs.pop("title", True)
        ytick = kwargs.pop("ytick", "")
        time_ticks = kwargs.pop("time_ticks", True)
        style_map = kwargs.pop("style_map", kwargs)  # Remaining kwargs become style_map

        return plot_segment(
            self, ax=ax, label_text=text, title=title, ytick=ytick, time_ticks=time_ticks, style_map=style_map
        )

    @classmethod
    def from_intervals(
        cls, intervals: np.ndarray, labels: list[str | None] | None = None, name: str | None = None
    ) -> "Segmentation":
        """Create segmentation from an interval array."""
        if labels is None:
            labels = [None] * len(intervals)
        time_spans = [TimeSpan(start=itvl[0], end=itvl[1], name=label) for itvl, label in zip(intervals, labels)]
        return cls(segments=time_spans, name=name)

    @classmethod
    def from_boundaries(
        cls, boundaries: list[float], labels: list[str | None] | None = None, name: str | None = None
    ) -> "Segmentation":
        """Create segmentation from a list of boundaries."""
        intervals = boundaries_to_intervals(np.array(sorted(boundaries)))
        return cls.from_intervals(intervals, labels, name)

    @classmethod
    def from_jams(cls, anno: "jams.Annotation") -> "Segmentation":
        """Create a Segmentation object from a JAMS annotation."""
        segments = [TimeSpan(start=obs.time, end=obs.time + obs.duration, name=obs.value) for obs in anno]
        return cls(segments=segments, name=anno.namespace)


@dataclass
class Hierarchy(TimeSpan):
    """A hierarchical structure of segmentations."""

    layers: list[Segmentation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Hierarchy must contain at least one layer.")

        # Set start/end from non-empty layers
        non_empty_layers = [layer for layer in self.layers if layer.segments]
        if non_empty_layers:
            self.start = min(layer.start for layer in non_empty_layers)
            self.end = max(layer.end for layer in non_empty_layers)

            # Validate all layers span the same time range
            for layer in non_empty_layers:
                if not (np.isclose(layer.start, self.start) and np.isclose(layer.end, self.end)):
                    raise ValueError(
                        f"All layers must span the same time range. "
                        f"Expected {self.start:.2f}-{self.end:.2f}s, "
                        f"got {layer.start:.2f}-{layer.end:.2f}s for '{layer.name}'."
                    )

        super().__post_init__()

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx]

    @property
    def intervals(self) -> list[np.ndarray]:
        """Interval arrays for all layers."""
        return [layer.intervals for layer in self.layers]

    @property
    def labels(self) -> list[list[str | None]]:
        """Label lists for all layers."""
        return [layer.labels for layer in self.layers]

    @property
    def boundaries(self) -> list[list[float]]:
        """Boundary lists for all layers."""
        return [layer.boundaries for layer in self.layers]

    def __repr__(self) -> str:
        name_str = f"name='{self.name}'" if self.name is not None else "name='None'"
        return f"Hierarchy({name_str}, {len(self)} layers, duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        return f"Hierarchy(name='{self.name}', {len(self)} layers, duration={self.duration:.2f}s)"

    def plot(self, ax: Axes | None = None, text: bool = True, **kwargs: Any) -> tuple[Figure | SubFigure, Axes]:
        """Plot the hierarchy with each layer in a separate subplot."""
        from .viz import label_style_dict

        figsize = kwargs.pop("figsize", None)
        n_layers = len(self.layers)

        fig, axes = plt.subplots(
            n_layers, 1, figsize=figsize or (6, 0.5 + 0.5 * n_layers), sharex=True, constrained_layout=True
        )
        if n_layers == 1:
            axes = [axes]

        for i, (layer, ax) in enumerate(zip(self.layers, axes)):
            layer.plot(
                ax=ax,
                style_map=label_style_dict(layer.labels),
                title=False,
                ytick=f"Level {i}",
                time_ticks=(i == n_layers - 1),
            )

        # We always have at least one axis since Hierarchy is guaranteed to have at least one layer
        axes[-1].set_xlabel("Time (s)")
        return fig, axes[-1]

    def plot_single_axis(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: bool = True,
        time_ticks: bool = True,
        layer_height: float = 0.8,
        layer_gap: float = 0.1,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Plot all layers on a single axis."""
        from .viz import label_style_dict

        n_layers = len(self.layers)

        fig: Figure | SubFigure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (10, 2 + 0.5 * n_layers))
        else:
            fig = ax.figure

        total_height = n_layers * (layer_height + layer_gap) - layer_gap
        current_y = total_height

        for layer in self.layers:
            layer_ymin = current_y - layer_height
            self._plot_layer_on_axis(ax, layer, label_style_dict, layer_ymin, layer_height, total_height)
            current_y -= layer_height + layer_gap

        if title and self.name:
            ax.set_title(self.name)

        ax.set_xlim(self.start, self.end) if self.start != self.end else ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(0, total_height)

        if time_ticks:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticks([])

        # Set y-ticks with layer names
        ytick_positions = [total_height - (i * (layer_height + layer_gap) + layer_height / 2) for i in range(n_layers)]
        ytick_labels = [
            layer.name if layer.name and layer.name.strip() else f"Level {n_layers - 1 - i}"
            for i, layer in enumerate(self.layers)
        ]

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(reversed(ytick_labels))

        return fig, ax

    def _plot_layer_on_axis(
        self,
        ax: Axes,
        layer: Segmentation,
        style_provider: Any,
        layer_ymin: float,
        layer_height: float,
        total_height: float,
    ) -> None:
        """Helper to plot a single layer on the axis."""
        style_map = style_provider(layer.labels)
        layer_ymax = layer_ymin + layer_height

        for span in layer.segments:
            span_style = style_map.get(span.name or "", {})

            ax.axvspan(
                span.start,
                span.end,
                ymin=layer_ymin / total_height,
                ymax=layer_ymax / total_height,
                alpha=span_style.get("alpha", 0.7),
                **span_style,
            )

            if span.name:
                text_offset = 0.005 * max(self.end - self.start, 0.1)
                ax.text(
                    span.start + text_offset,
                    layer_ymin + layer_height / 2,
                    span.name,
                    va="center",
                    ha="left",
                    fontsize=7,
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
                )

    @classmethod
    def from_jams(cls, jams_annotation: "jams.Annotation", name: str | None = None) -> "Hierarchy":
        """Create a Hierarchy from a JAMS multi_segment annotation."""
        if jams_annotation.namespace != "multi_segment":
            raise ValueError(f"Expected 'multi_segment' namespace, got '{jams_annotation.namespace}'")

        from jams.eval import hierarchy_flatten

        hier_intervals, hier_labels = hierarchy_flatten(jams_annotation)

        segmentations = []
        for i, (intervals, labels) in enumerate(zip(hier_intervals, hier_labels)):
            seg = Segmentation.from_intervals(np.array(intervals), labels, name=f"level_{i}")
            segmentations.append(seg)

        # Get hierarchy name from JAMS metadata if not provided
        if not name:
            annotator_meta = jams_annotation.annotation_metadata.annotator
            if annotator_meta and "name" in annotator_meta:
                name = annotator_meta["name"]

        return cls(layers=segmentations, name=name)

    @classmethod
    def from_json(cls, json_data: list[list[list[Any]]], name: str | None = None) -> "Hierarchy":
        """Create hierarchy from JSON structure (Adobe EST format)."""
        if not isinstance(json_data, list):
            raise ValueError("JSON data must be a list of layers.")

        segmentations = []
        for i, layer_data in enumerate(json_data):
            if not (isinstance(layer_data, list) and len(layer_data) == 2):
                raise ValueError(f"Layer {i} malformed. Expected [intervals, labels].")

            intervals_data, labels_data = layer_data
            if not (isinstance(intervals_data, list) and isinstance(labels_data, list)):
                raise ValueError(f"Layer {i}: intervals and labels must be lists.")

            if len(intervals_data) != len(labels_data):
                raise ValueError(f"Layer {i}: mismatched intervals/labels count.")

            segments = []
            for interval, label in zip(intervals_data, labels_data):
                # Handle nested intervals: [[[0,1]]] -> [[0,1]] -> [0,1]
                actual_interval = (
                    interval[0]
                    if (isinstance(interval, list) and len(interval) == 1 and isinstance(interval[0], list))
                    else interval
                )

                if not (isinstance(actual_interval, list) and len(actual_interval) == 2):
                    raise ValueError(f"Layer {i}: malformed interval {actual_interval}.")

                try:
                    start, end = float(actual_interval[0]), float(actual_interval[1])
                    segments.append(TimeSpan(start=start, end=end, name=str(label)))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Layer {i}: invalid interval {actual_interval}: {e}") from e

            segments.sort(key=lambda s: s.start)
            segmentations.append(Segmentation(segments=segments, name=f"layer_{i}"))

        return cls(layers=segmentations, name=name)
