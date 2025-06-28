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


@dataclass
class TimeSpan:
    """A labeled time span with start and end times.

    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    name : str, optional
        Label for this time span.
    """

    start: float = 0.0
    end: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        # Round time values to 4 decimal places for consistent JSON serialization
        self.start = np.round(self.start, 4)
        self.end = np.round(self.end, 4)

        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must be less than end time ({self.end})")

        if self.name is not None:
            self.name = str(self.name)
        # else: # If name is None, TimeSpan.__str__ will handle it. Or could set a default like ""
            # self.name = str(self) # This would make name default to the string representation of the TimeSpan itself

    def __str__(self) -> str:
        lab = self.name if self.name is not None else "" # Ensure lab is string
        return f"[{self.start:.1f}-{self.end:.1f}s]{lab}"

    def __repr__(self) -> str:
        return f"TimeSpan({self})"

    def plot(  # type: ignore[override]
        self,
        ax: Axes | None = None,
        text: bool = True,
        **style_map: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Plot the time span as a vertical bar."""
        if ax is None:
            _, ax = plt.subplots()

        # Convert color to facecolor to preserve edgecolor, default edgecolor to white
        if "color" in style_map:
            style_map["facecolor"] = style_map.pop("color")
        style_map.setdefault("edgecolor", "white")

        rect = ax.axvspan(self.start, self.end, **style_map)

        # Get ymax for annotation positioning, default to 1 (top of axes)
        span_ymax = style_map.get("ymax", 1.0)  # get the top of the rect span
        if text:
            lab_text = self.name if self.name is not None else ""
            ann = ax.annotate(
                lab_text,
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
    """A segmentation containing multiple time spans.

    Parameters
    ----------
    segments : list[TimeSpan]
        A sorted, non-overlapping, and contiguous list of `TimeSpan` objects.
    """

    segments: list[TimeSpan] = field(default_factory=list)

    def __post_init__(self) -> None:
        # order the segments by start time
        self.segments = sorted(self.segments, key=lambda x: x.start)
        # I should check that the segments are non-overlapping and contiguous.
        # Relax this check for common event-like namespaces where contiguity is not expected.
        event_namespaces = ["beat", "chord", "onset", "note"] # Add more if needed
        is_event_segmentation = self.name in event_namespaces

        if not is_event_segmentation:
            for i in range(len(self.segments) - 1):
                # Allow for small floating point inaccuracies
                if not np.isclose(self.segments[i].end, self.segments[i + 1].start):
                    raise ValueError(
                        f"Segments must be non-overlapping and contiguous. "
                        f"Gap found between segment {i} (end: {self.segments[i].end}) and "
                        f"segment {i+1} (start: {self.segments[i+1].start}). "
                        f"Segmentation name: '{self.name}'"
                    )

        # Set start/end from segments if available
        if self.segments:
            self.start = self.segments[0].start
            self.end = self.segments[-1].end

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx]

    @property
    def labels(self) -> list[str | None]:
        """A list of labels from all segments."""
        return [seg.name for seg in self.segments]

    @property
    def itvls(self) -> np.ndarray:
        """Intervals as an array of (start, end) pairs."""
        if not self.segments:
            return np.array([])
        return np.array([[seg.start, seg.end] for seg in self.segments])

    @property
    def bdrys(self) -> list[float]:
        """A sorted list of all boundary times."""
        if not self.segments:
            return []
        boundaries = [self.segments[0].start]
        boundaries.extend([seg.end for seg in self.segments])
        return boundaries

    def __repr__(self) -> str:
        dur = self.end - self.start
        return f"Segmentation({len(self)} segments over {dur:.2f}s)"

    def plot(  # type: ignore[override]
        self,
        ax: Axes | None = None,
        text: bool = True,
        title: bool = True,
        ytick: str = "",
        time_ticks: bool = True,
        style_map: dict[str, Any] | None = None,
    ) -> tuple[Figure | SubFigure, Axes]:
        """A convenience wrapper around `bnl.viz.plot_segment`."""
        # Local import to avoid circular dependency at module level
        from .viz import plot_segment

        return plot_segment(
            self,
            ax=ax,
            label_text=text,
            title=title,
            ytick=ytick,
            time_ticks=time_ticks,
            style_map=style_map,
        )

    def __str__(self) -> str:
        if len(self) == 0:
            return "Segmentation(0 segments): []"

        dur = self.end - self.start
        return f"Segmentation({len(self)} segments over {dur:.2f}s)"

    @classmethod
    def from_intervals(
        cls,
        intervals: np.ndarray,
        labels: list[str | None] | None = None,
        name: str | None = None,
    ) -> "Segmentation":
        """Create segmentation from an interval array.

        Parameters
        ----------
        intervals : np.ndarray, shape=(n, 2)
            An array of (start, end) times.
        labels : list of str, optional
            A list of labels for each interval.
        name : str, optional
            A name for the segmentation.
        """
        # Default labels is the interval string
        if labels is None:
            labels = [None] * len(intervals)

        time_spans = [TimeSpan(start=itvl[0], end=itvl[1], name=label) for itvl, label in zip(intervals, labels)]
        return cls(segments=time_spans, name=name)

    @classmethod
    def from_boundaries(
        cls,
        boundaries: list[float],
        labels: list[str | None] | None = None,
        name: str | None = None,
    ) -> "Segmentation":
        """Create segmentation from a list of boundaries.

        Parameters
        ----------
        boundaries : list of float
            A sorted list of boundary times.
        labels : list of str, optional
            A list of labels for each created segment.
        name : str, optional
            A name for the segmentation.
        """
        intervals = boundaries_to_intervals(np.array(sorted(boundaries)))
        return cls.from_intervals(intervals, labels, name)

    @classmethod
    def from_jams(cls, anno: jams.Annotation) -> "Segmentation":
        """Create a Segmentation object from a JAMS annotation.

        This method iterates through the observations of the provided JAMS
        annotation, creating a TimeSpan for each. It expects observations
        to have 'time', 'duration', and 'value' attributes. The resulting
        Segmentation object will have its `name` attribute set to the
        namespace of the input JAMS annotation (e.g., 'segment_open',
        'beat', 'chord').

        Parameters
        ----------
        anno : jams.Annotation
            The JAMS annotation object to convert.

        Returns
        -------
        Segmentation
            A new Segmentation object.
        """
        segments = []
        for obs in anno:
            segments.append(TimeSpan(start=obs.time, end=obs.time + obs.duration, name=obs.value))
        # Use the annotation's namespace as the name for the Segmentation object
        return cls(segments=segments, name=anno.namespace)


@dataclass
class Hierarchy(TimeSpan):
    """A hierarchical structure of segmentations.

    Parameters
    ----------
    layers : list[Segmentation]
        An ordered list of `Segmentation` objects, from coarsest to finest.
    """

    layers: list[Segmentation] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Set start/end from layers if available
        if self.layers:
            # Find the first non-empty layer to get start/end times
            non_empty_layer = next((layer for layer in self.layers if layer.segments), None)
            if non_empty_layer:
                self.start = non_empty_layer.start
                self.end = non_empty_layer.end

        # Check that all non-empty layers have the same start and end time
        for layer in self.layers:
            if layer.segments and (layer.start != self.start or layer.end != self.end):
                raise ValueError("All layers must have the same start and end time.")

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx]

    @property
    def itvls(self) -> list[np.ndarray]:
        """A list of interval arrays for all levels."""
        return [lvl.itvls for lvl in self.layers]

    @property
    def labels(self) -> list[list[str | None]]:
        """A list of label lists for all levels."""
        return [lvl.labels for lvl in self.layers]

    @property
    def bdrys(self) -> list[list[float]]:
        """A list of boundary lists for all levels."""
        return [lvl.bdrys for lvl in self.layers]

    def __repr__(self) -> str:
        return f"Hierarchy({len(self)} levels over {self.start:.2f}s-{self.end:.2f}s)"

    def __str__(self) -> str:
        if len(self) == 0:
            return "Hierarchy(0 levels)"

        return f"Hierarchy({len(self)} levels over {self.start:.2f}s-{self.end:.2f}s)"

    def plot(  # type: ignore[override]
        self,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot the hierarchy with each layer in a separate subplot.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height)
        Returns
        -------
        fig : matplotlib figure
        """
        from .viz import label_style_dict

        n_layers = len(self.layers)
        if n_layers == 0:
            raise ValueError("Cannot plot empty hierarchy")

        # Create subplots - one for each layer
        fig, axes = plt.subplots(
            n_layers, 1, figsize=figsize or (6, 0.5 + 0.5 * n_layers), sharex=True, constrained_layout=True
        )
        if n_layers == 1:
            axes = [axes]

        # Plot each layer using Segmentation.plot()
        for i, (layer, ax) in enumerate(zip(self.layers, axes)):
            layer.plot(
                ax=ax,
                style_map=label_style_dict(layer.labels) if len(layer) > 0 else None,
                title=False,
                ytick=f"Level {i}",
                time_ticks=(i == (n_layers - 1)),
            )
        # Set x-label only on bottom subplot
        axes[-1].set_xlabel("Time (s)")
        return fig

    @classmethod
    def from_jams(cls, jams_annotation: "jams.Annotation") -> "Hierarchy":
        """Create a Hierarchy from a JAMS multi_segment annotation.

        Parameters
        ----------
        jams_annotation : jams.Annotation
            A JAMS annotation with namespace 'multi_segment'

        Returns
        -------
        Hierarchy
            A new Hierarchy object with multiple segmentation layers
        """
        if jams_annotation.namespace != "multi_segment":
            raise ValueError(f"Expected 'multi_segment' namespace, got '{jams_annotation.namespace}'")

        # Use JAMS' built-in hierarchy flattening function
        from jams.eval import hierarchy_flatten

        hier_intervals, hier_labels = hierarchy_flatten(jams_annotation)

        # Convert each level to a Segmentation using existing constructor
        segmentations = []
        for intervals, labels in zip(hier_intervals, hier_labels):
            seg = Segmentation.from_intervals(np.array(intervals), labels)
            segmentations.append(seg)

        return cls(layers=segmentations)

    @classmethod
    def from_json(cls, json_data: list[list[list[Any]]], name: str | None = None) -> "Hierarchy":
        """Create hierarchy from a JSON annotation (Adobe EST format).

        The JSON data is expected to be a list of layers. Each layer is a list
        containing two sub-lists:
        1. A list of intervals, where each interval is `[start_time, end_time]`.
        2. A list of corresponding labels for these intervals.

        Example structure:
        [
            [[[0.0, 10.0]], ["A"]],  # Layer 0
            [[[0.0, 5.0], [5.0, 10.0]], ["a", "b"]]  # Layer 1
        ]

        Parameters
        ----------
        json_data : list[list[list[Any]]]
            The Adobe EST JSON data.
        name : str, optional
            A name for the hierarchy.

        Returns
        -------
        Hierarchy
            A new Hierarchy object.
        """
        segmentations = []
        if not isinstance(json_data, list):
            raise ValueError("JSON data must be a list of layers.")

        for i, layer_data in enumerate(json_data):
            if not (isinstance(layer_data, list) and len(layer_data) == 2):
                raise ValueError(f"Layer {i} is malformed. Expected a list of [intervals, labels], got: {layer_data}")

            intervals_data, labels_data = layer_data

            if not (isinstance(intervals_data, list) and isinstance(labels_data, list)):
                raise ValueError(
                    f"Layer {i} intervals or labels are not lists. Got intervals: {type(intervals_data)}, labels: {type(labels_data)}"
                )

            if len(intervals_data) != len(labels_data):
                raise ValueError(
                    f"Layer {i} has mismatched number of intervals and labels. "
                    f"{len(intervals_data)} intervals, {len(labels_data)} labels."
                )

            segments = []
            for interval, label in zip(intervals_data, labels_data):
                # Accommodate for intervals possibly being nested e.g. [[[0.0, 10.0]]] vs [[0.0, 10.0]]
                actual_interval = interval
                if (
                    isinstance(interval, list)
                    and len(interval) == 1
                    and isinstance(interval[0], list)
                    and len(interval[0]) == 2
                ):
                    actual_interval = interval[0]

                if not (isinstance(actual_interval, list) and len(actual_interval) == 2):
                    raise ValueError(
                        f"Malformed interval structure in layer {i}: {actual_interval}. Expected [start, end]."
                    )

                start, end = actual_interval
                try:
                    segments.append(TimeSpan(start=float(start), end=float(end), name=str(label)))
                except ValueError as e:  # Catch errors from float conversion or TimeSpan post_init
                    raise ValueError(
                        f"Error creating TimeSpan for interval {actual_interval} with label '{label}' in layer {i}: {e}"
                    )

            segments.sort(key=lambda s: s.start)
            segmentations.append(Segmentation(segments=segments))

        return cls(layers=segmentations, name=name)

    def plot_single_axis(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: bool = True,
        time_ticks: bool = True,
        layer_height: float = 0.8,
        layer_gap: float = 0.1,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Plot all layers of the hierarchy on a single axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing axes to plot on.
        figsize : tuple, optional
            Figure size (width, height) if creating a new figure.
        title : bool, default=True
            Whether to display the hierarchy's name as a title.
        time_ticks : bool, default=True
            Whether to display time ticks on the x-axis.
        layer_height : float, default=0.8
            The vertical height allocated to each layer's segments on the axis.
        layer_gap : float, default=0.1
            The vertical gap between layers on the axis.

        Returns
        -------
        fig : matplotlib.figure.Figure or SubFigure
        ax : matplotlib.axes.Axes
        """
        from .viz import label_style_dict  # Local import for viz functions

        n_layers = len(self.layers)
        if n_layers == 0:
            raise ValueError("Cannot plot empty hierarchy")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (10, 2 + 0.5 * n_layers))
        else:
            fig = ax.figure

        # Calculate total height needed for all layers for y-axis scaling of axvspan
        total_plot_height = n_layers * (layer_height + layer_gap) - layer_gap
        if total_plot_height <= 0:
            total_plot_height = 1  # Avoid division by zero for empty plots

        current_y_base = total_plot_height  # Start plotting from the top

        for i, layer in enumerate(self.layers):
            # Calculate ymin and ymax for the current layer's segments
            # These are absolute data coordinates for the ax.text, but relative for axvspan
            layer_ymin_abs = current_y_base - layer_height
            layer_ymax_abs = current_y_base

            if layer.segments:  # Check if layer has segments
                style_map = label_style_dict(layer.labels)
                for span in layer.segments:
                    span_style = style_map.get(span.name if span.name is not None else "", {})

                    # axvspan ymin/ymax are relative to the axes (0-1)
                    rect_ymin_rel = layer_ymin_abs / total_plot_height
                    rect_ymax_rel = layer_ymax_abs / total_plot_height

                    ax.axvspan(
                        span.start,
                        span.end,
                        ymin=rect_ymin_rel,
                        ymax=rect_ymax_rel,
                        **span_style,
                        alpha=span_style.get("alpha", 0.7),
                    )

                    if span.name:
                        ax.text(
                            span.start + 0.005 * (self.end - self.start),
                            layer_ymin_abs + layer_height / 2,
                            f"{span.name}",
                            va="center",
                            ha="left",
                            fontsize=7,
                            clip_on=True,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
                        )

            current_y_base -= layer_height + layer_gap

        if title and self.name:
            ax.set_title(self.name)

        if self.start != self.end:
            ax.set_xlim(self.start, self.end)
        else:
            ax.set_xlim(-0.1, 0.1)

        ax.set_ylim(0, total_plot_height)  # Y-axis uses data coordinates reflecting the structure

        if time_ticks:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticks([])

        # Set y-ticks to correspond to the center of each layer's band
        ytick_positions = [
            total_plot_height - (i * (layer_height + layer_gap) + layer_height / 2) for i in range(n_layers)
        ]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([f"Level {i}" for i in reversed(range(n_layers))])

        return fig, ax

    @classmethod
    def from_boundaries(
        cls, boundaries: list[list[float]], labels: list[list[str] | None] | None = None, name: str | None = None
    ) -> "Hierarchy":
        raise NotImplementedError

    @classmethod
    def from_intervals(
        cls, intervals: list[np.ndarray], labels: list[list[str] | None] | None = None, name: str | None = None
    ) -> "Hierarchy":
        raise NotImplementedError
