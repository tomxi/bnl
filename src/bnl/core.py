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
        _s = self.start
        _e = self.end

        # Explicitly handle if _s or _e are empty lists, defaulting to 0.0
        if isinstance(_s, list) and not _s:
            _s = 0.0
        if isinstance(_e, list) and not _e:
            _e = 0.0

        # Ensure they are floatable before rounding
        try:
            _s = float(_s)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"TimeSpan start must be convertible to float, got {self.start} (type {type(self.start)})"
            ) from e
        try:
            _e = float(_e)
        except (TypeError, ValueError) as e:
            raise TypeError(f"TimeSpan end must be convertible to float, got {self.end} (type {type(self.end)})") from e

        self.start = np.round(_s, 4)
        self.end = np.round(_e, 4)

        if self.start < 0:
            raise ValueError(f"Start time ({self.start}) cannot be negative.")
        if self.end < 0:
            raise ValueError(f"End time ({self.end}) cannot be negative.")
        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}) must be less than or equal to end time ({self.end}).")

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
        super().__post_init__()  # Call TimeSpan's __post_init__ for start/end validation
        # Order the segments by start time
        self.segments = sorted(self.segments, key=lambda x: x.start)

        # Determine if this segmentation should enforce contiguity
        # Common event-like namespaces where contiguity is not expected
        event_namespaces = ["beat", "chord", "onset", "note", "key", "tempo", "lyrics", "chroma"]
        is_event_segmentation = self.name is not None and any(
            event_ns in self.name.lower() for event_ns in event_namespaces
        )

        if not is_event_segmentation and self.segments:
            for i in range(len(self.segments) - 1):
                # Allow for small floating point inaccuracies (e.g., 1e-9)
                if not np.isclose(self.segments[i].end, self.segments[i + 1].start, atol=1e-9):
                    raise ValueError(
                        f"Segments must be contiguous for this segmentation type. "
                        f"Gap found between segment {i} (end: {self.segments[i].end}) and "
                        f"segment {i + 1} (start: {self.segments[i + 1].start}). "
                        f"Segmentation name: '{self.name}'"
                    )
            # Check for overlaps
            for i in range(len(self.segments) - 1):
                if self.segments[i].end > self.segments[i + 1].start:
                    if not np.isclose(
                        self.segments[i].end, self.segments[i + 1].start, atol=1e-9
                    ):  # allow for minor overlaps due to rounding
                        raise ValueError(
                            f"Segments must be non-overlapping. "
                            f"Overlap found between segment {i} (end: {self.segments[i].end}) and "
                            f"segment {i + 1} (start: {self.segments[i + 1].start}). "
                            f"Segmentation name: '{self.name}'"
                        )

        # Set overall start/end from segments if segments are present
        if not self.segments:
            raise ValueError("Segmentation must contain at least one segment.")

        self.start = self.segments[0].start
        self.end = self.segments[-1].end
        # If segments is empty, TimeSpan's __post_init__ already handled start/end (e.g. 0.0, 0.0)
        # No need to call super().__post_init__() again here as it might overwrite if called late

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx]

    @property
    def labels(self) -> list[str | None]:
        """A list of labels from all segments. Returns an empty list if no segments."""
        return [seg.name for seg in self.segments]

    @property
    def intervals(self) -> np.ndarray:
        """Intervals as a NumPy array of (start, end) pairs.
        Returns an empty array of shape (0, 2) if no segments.
        """
        if not self.segments:
            return np.empty((0, 2))
        return np.array([[seg.start, seg.end] for seg in self.segments])

    @property
    def boundaries(self) -> list[float]:
        """A sorted list of unique boundary times (segment starts and ends).
        Returns an empty list if no segments.
        """
        if not self.segments:
            return []
        # Collect all start and end times
        b = {seg.start for seg in self.segments} | {seg.end for seg in self.segments}
        return sorted(list(b))

    def __repr__(self) -> str:
        name_str = f"name='{self.name}', " if self.name else ""
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

    def plot(  # type: ignore[override]
        self,
        ax: Axes | None = None,
        text: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure | SubFigure, Axes]:
        """A convenience wrapper around `bnl.viz.plot_segment`.

        Compatible with TimeSpan.plot, accepts additional kwargs for customization.
        Relevant kwargs: title (bool), ytick (str), time_ticks (bool), style_map (dict).
        """
        # Local import to avoid circular dependency at module level
        from .viz import plot_segment

        # Extract specific parameters from kwargs with defaults
        title = kwargs.pop("title", True)
        ytick = kwargs.pop("ytick", "")
        time_ticks = kwargs.pop("time_ticks", True)
        style_map_from_kwargs = kwargs.pop("style_map", None)

        # Remaining kwargs are treated as style_map, but explicit style_map takes precedence
        # This matches TimeSpan.plot's **style_map behavior if no explicit 'style_map' kwarg is given.
        # If 'style_map' is provided in kwargs, it's used. Otherwise, kwargs itself is the style_map.
        # However, plot_segment expects style_map as a specific dict.
        # For TimeSpan compatibility, kwargs directly become style_map items.
        # For Segmentation's plot_segment, we need to be careful.
        # If style_map_from_kwargs is provided, use it. Otherwise, use remaining kwargs as style_map.

        final_style_map = style_map_from_kwargs if style_map_from_kwargs is not None else kwargs

        return plot_segment(
            self,
            ax=ax,
            label_text=text,  # from base signature
            title=title,  # from kwargs
            ytick=ytick,  # from kwargs
            time_ticks=time_ticks,  # from kwargs
            style_map=final_style_map,  # from kwargs or explicit style_map kwarg
        )

    def __str__(self) -> str:
        # Align with __repr__ for consistency, including name and duration format
        name_str = f"name='{self.name}', " if self.name else ""
        if not self.segments:  # Handles empty segmentation
            return f"Segmentation({name_str}0 segments, duration={self.duration:.2f}s)"
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

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
        super().__post_init__()  # Call TimeSpan's __post_init__

        if not self.layers:
            raise ValueError("Hierarchy must contain at least one layer.")

        # Determine overall start and end times from the layers
        # If layers is empty, self.start and self.end remain as initialized by TimeSpan (e.g., 0.0, 0.0)
        if self.layers:  # This if self.layers is now redundant due to the check above, but harmless
            # Filter out empty segmentations before finding min/max times
            non_empty_layers = [layer for layer in self.layers if layer.segments]
            if non_empty_layers:
                self.start = min(layer.start for layer in non_empty_layers)
                self.end = max(layer.end for layer in non_empty_layers)

                # Check that all non-empty layers have the same start and end time as the hierarchy
                for layer in non_empty_layers:
                    if not np.isclose(layer.start, self.start) or not np.isclose(layer.end, self.end):
                        layer_idx = self.layers.index(layer)
                        err_msg = (
                            f"All non-empty layers in a Hierarchy must span the same overall time range. "
                            f"Hierarchy: {self.start:.2f}s-{self.end:.2f}s. "
                            f"Layer '{layer.name}' (index {layer_idx}): {layer.start:.2f}s-{layer.end:.2f}s."
                        )
                        raise ValueError(err_msg)
            # If all layers are empty segmentations, self.start/end are already set (e.g. to 0.0, 0.0 by TimeSpan)
            # and no further validation on layer times is needed.

        # Ensure layers are ordered (though typically they are constructed in order)
        # This doesn't sort by any specific criteria other than their initial list order.
        # If a specific sorting (e.g., by number of segments) is needed, it should be explicit.

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx]

    @property
    def intervals(self) -> list[np.ndarray]:
        """A list of interval arrays (NumPy) for all layers.
        Each element in the list corresponds to a layer's intervals.
        """
        return [lvl.intervals for lvl in self.layers]

    @property
    def labels(self) -> list[list[str | None]]:
        """A list of label lists for all layers.
        Each sub-list contains labels for segments in the corresponding layer.
        """
        return [lvl.labels for lvl in self.layers]

    @property
    def boundaries(self) -> list[list[float]]:
        """A list of boundary lists (unique, sorted times) for all layers."""
        return [lvl.boundaries for lvl in self.layers]

    def __repr__(self) -> str:
        name_str = f"name='{self.name}'" if self.name is not None else "name='None'"
        return f"Hierarchy({name_str}, {len(self)} layers, duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        if not self.layers:
            return f"Hierarchy(name='{self.name}', 0 layers, duration={self.duration:.2f}s)"
        return f"Hierarchy(name='{self.name}', {len(self)} layers, duration={self.duration:.2f}s)"

    def plot(  # type: ignore[override]
        self,
        ax: Axes | None = None,  # Ignored, but needed for signature compatibility
        text: bool = True,  # Ignored, but needed for signature compatibility
        **kwargs: Any,
    ) -> Figure:
        """Plot the hierarchy with each layer in a separate subplot.

        Compatible with TimeSpan.plot signature by accepting ax, text, and **kwargs.
        The `figsize` keyword argument (a tuple) is used; other keyword arguments are ignored.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height). Extracted from kwargs.
        ax : Axes, optional
            Ignored for this plotting method. Included for signature compatibility.
        text : bool, optional
            Ignored for this plotting method. Included for signature compatibility.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Code of the function starts here
        figsize = kwargs.pop("figsize", None)
        # ax and text parameters are accepted for signature compatibility but ignored here.
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
        if n_layers > 0:  # Ensure axes is not empty if hierarchy has no layers (though plot would error earlier)
            axes[-1].set_xlabel("Time (s)")
        return fig

    @classmethod
    def from_jams(cls, jams_annotation: "jams.Annotation", name: str | None = None) -> "Hierarchy":
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
        # Assign default names to segmentations if hierarchy_flatten doesn't provide them
        # Or, consider modifying Segmentation.from_intervals to accept a default name pattern
        for i, (intervals, labels) in enumerate(zip(hier_intervals, hier_labels)):
            seg_name = f"level_{i}"  # Default name, could be improved if JAMS provides layer names
            seg = Segmentation.from_intervals(np.array(intervals), labels, name=seg_name)
            segmentations.append(seg)

        hierarchy_name = name
        if not hierarchy_name:
            annotator_meta = jams_annotation.annotation_metadata.annotator
            if annotator_meta and "name" in annotator_meta:
                hierarchy_name = annotator_meta["name"]
            elif hasattr(jams_annotation, "file_metadata") and jams_annotation.file_metadata.title:
                hierarchy_name = jams_annotation.file_metadata.title  # Fallback

        return cls(layers=segmentations, name=hierarchy_name)

    @classmethod
    def from_json(cls, json_data: list[list[list[Any]]], name: str | None = None) -> "Hierarchy":
        """Create hierarchy from a JSON-like structure (e.g., Adobe EST format).

        The JSON data is expected to be a list of layers. Each layer is a list
        containing two sub-lists:
        1. A list of intervals, where each interval is `[start_time, end_time]`.
        2. A list of corresponding labels for these intervals.

        Example structure::

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
                    f"Layer {i} intervals or labels are not lists. "
                    f"Got intervals: {type(intervals_data)}, labels: {type(labels_data)}"
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
                    ) from e

            segments.sort(key=lambda s: s.start)
            # Assign a default name to the segmentation layer if not otherwise specified
            layer_name = f"layer_{i}"  # Consider if a more descriptive name can be derived
            segmentations.append(Segmentation(segments=segments, name=layer_name))

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

        fig: Figure | SubFigure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (10, 2 + 0.5 * n_layers))
        else:
            fig = ax.figure

        # Calculate total height needed for all layers for y-axis scaling of axvspan
        total_plot_height = n_layers * (layer_height + layer_gap) - layer_gap
        if total_plot_height <= 0:
            total_plot_height = 1  # Avoid division by zero for empty plots

        current_y_base = total_plot_height  # Start plotting from the top

        for _i, layer in enumerate(self.layers):
            layer_ymin_abs = current_y_base - layer_height
            self._plot_layer_segments_on_axis(
                ax,
                layer,
                style_map_provider=label_style_dict,
                layer_ymin_abs=layer_ymin_abs,
                layer_height=layer_height,
                total_plot_height=total_plot_height,
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
        # Also, use layer names if available, otherwise default to "Level i"
        ytick_labels = []
        ytick_positions = []
        for i, layer in enumerate(self.layers):
            ytick_positions.append(total_plot_height - (i * (layer_height + layer_gap) + layer_height / 2))
            layer_display_name = (
                layer.name if layer.name and layer.name.strip() else f"Level {len(self.layers) - 1 - i}"
            )
            ytick_labels.append(layer_display_name)

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(reversed(ytick_labels))  # Reversed because plotting order is top-to-bottom

        return fig, ax

    def _plot_layer_segments_on_axis(
        self,
        ax: Axes,
        layer: Segmentation,
        style_map_provider: Any,  # Actually Callable[[list[str|None]], dict[str, Any]]
        layer_ymin_abs: float,
        layer_height: float,
        total_plot_height: float,
    ) -> None:
        """Helper to plot segments of a single layer onto the provided axis."""
        if not layer.segments:
            return

        style_map = style_map_provider(layer.labels)
        layer_ymax_abs = layer_ymin_abs + layer_height

        for span in layer.segments:
            span_style = style_map.get(span.name if span.name is not None else "", {})

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
                # Calculate a small offset for text based on overall hierarchy duration
                # to avoid text being too close to the start if duration is very small.
                text_x_offset = 0.005 * (self.end - self.start) if self.end > self.start else 0.005
                ax.text(
                    span.start + text_x_offset,
                    layer_ymin_abs + layer_height / 2,
                    f"{span.name}",
                    va="center",
                    ha="left",
                    fontsize=7,
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
                )

    # @classmethod
    # def from_boundaries(
    #     cls, boundaries: list[list[float]], labels: list[list[str] | None] | None = None, name: str | None = None
    # ) -> "Hierarchy":
    #     """Create a Hierarchy from lists of boundaries for each layer."""
    #     layers = []
    #     for i, layer_boundaries in enumerate(boundaries):
    #         layer_labels = labels[i] if labels and i < len(labels) else None
    #         layer_name = f"layer_{i}" # Default name
    #         # Potentially pass a more specific name if available from `name` or other source
    #         layers.append(Segmentation.from_boundaries(layer_boundaries, layer_labels, name=layer_name))
    #     return cls(layers=layers, name=name)

    # @classmethod
    # def from_intervals(
    #     cls,
    #     intervals_per_layer: list[np.ndarray],
    #     labels_per_layer: list[list[str | None] | None] | None = None,
    #     name: str | None = None,
    # ) -> "Hierarchy":
    #     """Create a Hierarchy from lists of interval arrays for each layer."""
    #     layers = []
    #     for i, layer_intervals in enumerate(intervals_per_layer):
    #         layer_labels = labels_per_layer[i] if labels_per_layer and i < len(labels_per_layer) else None
    #         layer_name = f"layer_{i}" # Default name
    #         layers.append(Segmentation.from_intervals(layer_intervals, layer_labels, name=layer_name))
    #     return cls(layers=layers, name=name)
