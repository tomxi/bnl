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
        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must be less than end time ({self.end})")
        if self.name is None:
            self.name = str(self)

    def __str__(self) -> str:
        lab = self.name if self.name else ""
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
        for i in range(len(self.segments) - 1):
            if self.segments[i].end != self.segments[i + 1].start:
                raise ValueError("Segments must be non-overlapping and contiguous.")

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
        """Create segmentation from a JAMS annotation. (Not yet implemented)"""
        # TODO: Implement JAMS open_segment annotation parsing
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "Segmentation":
        """Create segmentation from a JAMS annotation. (Not yet implemented)"""
        # TODO: Implement JSON open_segment annotation parsing
        raise NotImplementedError


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
            self.start = self.layers[0].start
            self.end = self.layers[0].end

        for layer in self.layers:
            if layer.start != self.start or layer.end != self.end:
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
        self, figsize: tuple[float, float] = (8, 6), layer_height: float = 1.0, layer_spacing: float = 0.2
    ) -> tuple:
        """Plot the hierarchy with each layer in a separate subplot.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height)
        layer_height : float, optional
            Height of each layer in the plot
        layer_spacing : float, optional
            Spacing between layers

        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        from .viz import label_style_dict

        n_layers = len(self.layers)
        if n_layers == 0:
            raise ValueError("Cannot plot empty hierarchy")

        # Create subplots - one for each layer
        fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
        if n_layers == 1:
            axes = [axes]

        # Plot each layer using Segmentation.plot()
        for i, (layer, ax) in enumerate(zip(self.layers, axes)):
            # Create style map for this layer with ymin/ymax for positioning
            if len(layer) > 0:
                style_map = label_style_dict(layer.labels)
                # Add ymin/ymax to each style to position segments properly
                for label_style in style_map.values():
                    label_style.update({"ymin": 0, "ymax": layer_height})
            else:
                style_map = None

            # Use Segmentation.plot() which will call TimeSpan.plot() for each segment
            layer.plot(ax=ax, style_map=style_map, title=False, ytick=f"Level {i}")

            # Set y limits for this subplot
            ax.set_ylim(0, layer_height)

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

        # Group observations by level
        levels: dict[int, list[tuple[float, float, str]]] = {}
        for obs in jams_annotation.data:
            level = obs.value["level"]
            label = obs.value["label"]

            if level not in levels:
                levels[level] = []
            levels[level].append((obs.time, obs.time + obs.duration, label))

        # Create Segmentation objects for each level
        segmentations = []
        for level in sorted(levels.keys()):
            # Extract boundaries and labels for this level
            intervals = levels[level]
            boundaries = sorted(
                set([start for start, end, label in intervals] + [end for start, end, label in intervals])
            )

            # Create labels list corresponding to segments between boundaries
            labels: list[str | None] = []
            for i in range(len(boundaries) - 1):
                seg_start, seg_end = boundaries[i], boundaries[i + 1]
                # Find the label for this segment
                for start, end, label in intervals:
                    if start <= seg_start and seg_end <= end:
                        labels.append(label)
                        break
                else:
                    labels.append(None)  # No label found for this segment

            seg = Segmentation.from_boundaries(boundaries, labels)
            segmentations.append(seg)

        return cls(layers=segmentations)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "Hierarchy":
        """Create hierarchy from a JSON annotation. (Not yet implemented)"""
        # TODO: Implement JSON multilevel annotation parsing
        raise NotImplementedError
