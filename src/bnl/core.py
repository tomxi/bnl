"""Core data structures and constructors."""

from dataclasses import dataclass, field
from typing import Any, List, Tuple # Added List, Tuple

import jams
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from mir_eval.util import boundaries_to_intervals

__all__ = ["TimeSpan", "Segmentation", "Hierarchy", "RatedBoundaries", "ProperHierarchy"]


def _validate_time_value(value: Any, name: str) -> float:
    """Convert and validate a time value."""
    if isinstance(value, list) and not value: # type: ignore
        value = 0.0
    try:
        result = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to float, got {value} (type {type(value)})") from e

    result = np.round(result, 4)
    if result < 0:
        raise ValueError(f"{name} ({result}) cannot be negative.")
    return result


def _check_segments_contiguous(segments: Tuple["TimeSpan", ...], name: str | None) -> None: # Changed to Tuple
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


@dataclass(frozen=True) # Added frozen=True
class TimeSpan:
    """A labeled time span with start and end times."""

    start: float = 0.0
    end: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        # Use object.__setattr__ for frozen dataclasses
        object.__setattr__(self, 'start', _validate_time_value(self.start, "Start time"))
        object.__setattr__(self, 'end', _validate_time_value(self.end, "End time"))

        if self.start > self.end: # Validation after setting
            raise ValueError(f"Start time ({self.start}) must be ≤ end time ({self.end}).")

        if self.name is not None:
            object.__setattr__(self, 'name', str(self.name))

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


@dataclass(frozen=True) # Added frozen=True
class Segmentation(TimeSpan):
    """A segmentation containing multiple time spans."""

    segments: List[TimeSpan] = field(default_factory=list) # Input can be List

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("Segmentation must contain at least one segment.")

        # Sort segments and store as a tuple to ensure immutability of the collection
        sorted_segments_tuple = tuple(sorted(self.segments, key=lambda x: x.start))
        object.__setattr__(self, 'segments', sorted_segments_tuple)

        # self.name is from TimeSpan's fields, potentially set by its __post_init__
        # _check_segments_contiguous expects self.name to be available.
        # TimeSpan's __post_init__ is called via super()

        seg_start = self.segments[0].start
        seg_end = self.segments[-1].end

        # Call super's __post_init__ to handle initialization of TimeSpan fields (start, end, name)
        # It will use the values passed to Segmentation's constructor for start, end, name, or defaults.
        super().__post_init__()

        # Now override start and end with values derived from segments
        object.__setattr__(self, 'start', _validate_time_value(seg_start, "Segmentation start time"))
        object.__setattr__(self, 'end', _validate_time_value(seg_end, "Segmentation end time"))

        # After overriding start/end, re-check contiguity with the final name
        # (name would have been set by super().__post_init__ if passed, or is None)
        _check_segments_contiguous(self.segments, self.name)


        # Final validation for TimeSpan start/end after potential override
        if self.start > self.end: # This check is also in TimeSpan's __post_init__
            raise ValueError(f"Derived start time ({self.start}) must be ≤ derived end time ({self.end}).")


    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx] # segments is now a tuple

    @property
    def labels(self) -> Tuple[str | None, ...]: # Return tuple for immutability
        return tuple(seg.name for seg in self.segments)

    @property
    def intervals(self) -> np.ndarray: # np.ndarray is mutable, but common for this type of data
        return np.array([[seg.start, seg.end] for seg in self.segments])

    @property
    def boundaries(self) -> Tuple[float, ...]: # Return tuple for immutability
        boundaries_set = {seg.start for seg in self.segments} | {seg.end for seg in self.segments}
        return tuple(sorted(list(boundaries_set)))

    def __repr__(self) -> str:
        name_str = f"name='{self.name}', " if self.name else ""
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        name_str = f"name='{self.name}', " if self.name else ""
        return f"Segmentation({name_str}{len(self)} segments, duration={self.duration:.2f}s)"

    def plot(self, ax: Axes | None = None, text: bool = True, **kwargs: Any) -> tuple[Figure | SubFigure, Axes]:
        from .viz import plot_segment # type: ignore
        title = kwargs.pop("title", True)
        ytick = kwargs.pop("ytick", "")
        time_ticks = kwargs.pop("time_ticks", True)
        style_map = kwargs.pop("style_map", kwargs)
        return plot_segment( # type: ignore
            self, ax=ax, label_text=text, title=title, ytick=ytick, time_ticks=time_ticks, style_map=style_map
        )

    @classmethod
    def from_intervals(
        cls, intervals: np.ndarray, labels: List[str | None] | None = None, name: str | None = None
    ) -> "Segmentation":
        if labels is None:
            labels = [None] * len(intervals)
        time_spans = [TimeSpan(start=itvl[0], end=itvl[1], name=label) for itvl, label in zip(intervals, labels)]
        # Pass name, segments. TimeSpan's start/end will be default or derived in __post_init__.
        return cls(name=name, segments=time_spans)

    @classmethod
    def from_boundaries(
        cls, boundaries: List[float], labels: List[str | None] | None = None, name: str | None = None
    ) -> "Segmentation":
        sorted_boundaries = sorted(list(set(boundaries))) # Ensure unique and sorted

        if not sorted_boundaries:
            # This case implies creating an empty segmentation, which is not allowed by __post_init__.
            # Raise here to make it explicit.
            raise ValueError("Cannot create Segmentation from an empty list of boundaries.")

        if len(sorted_boundaries) == 1:
            # Single boundary point, create a zero-duration segment at that point.
            ts = sorted_boundaries[0]
            label = labels[0] if labels and len(labels) > 0 else None
            time_spans = [TimeSpan(start=ts, end=ts, name=label)]
            # For cls call, ensure TimeSpan's start/end are not overridden by Segmentation's __post_init__
            # if they are not meant to be. Here, Segmentation's start/end will be ts,ts.
            return cls(name=name, segments=time_spans)
        else:
            # Multiple boundaries, use mir_eval
            intervals_arr = boundaries_to_intervals(np.array(sorted_boundaries))

            final_labels: List[str|None] | None = None
            if labels is not None:
                if len(labels) != len(intervals_arr): # Check against number of intervals
                    raise ValueError(
                        f"Number of labels ({len(labels)}) must match number of segments ({len(intervals_arr)})."
                    )
                final_labels = labels
            # If labels is None, from_intervals will create [None]*num_intervals

            return cls.from_intervals(intervals_arr, labels=final_labels, name=name)

    @classmethod
    def from_jams(cls, anno: "jams.Annotation") -> "Segmentation": # type: ignore
        segments = [TimeSpan(start=obs.time, end=obs.time + obs.duration, name=obs.value) for obs in anno] # type: ignore
        return cls(name=anno.namespace, segments=segments) # type: ignore


@dataclass(frozen=True) # Added frozen=True
class Hierarchy(TimeSpan):
    """A hierarchical structure of segmentations."""

    layers: List[Segmentation] = field(default_factory=list) # Input can be List

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Hierarchy must contain at least one layer.")

        # Convert layers to tuple for immutability
        object.__setattr__(self, 'layers', tuple(self.layers))

        hier_start: float | None = None
        hier_end: float | None = None

        non_empty_layers = [layer for layer in self.layers if layer.segments]
        if non_empty_layers:
            hier_start = min(layer.start for layer in non_empty_layers)
            hier_end = max(layer.end for layer in non_empty_layers)

            for layer in non_empty_layers:
                if not (np.isclose(layer.start, hier_start) and np.isclose(layer.end, hier_end)): # type: ignore
                    raise ValueError(
                        f"All layers must span the same time range. "
                        f"Expected {hier_start:.2f}-{hier_end:.2f}s, " # type: ignore
                        f"got {layer.start:.2f}-{layer.end:.2f}s for '{layer.name}'."
                    )

        super().__post_init__() # Initialize TimeSpan fields (name, and default/passed start/end)

        if hier_start is not None and hier_end is not None:
            object.__setattr__(self, 'start', _validate_time_value(hier_start, "Hierarchy start time"))
            object.__setattr__(self, 'end', _validate_time_value(hier_end, "Hierarchy end time"))
        # If non_empty_layers is empty, TimeSpan's start/end (either default 0.0 or passed values) remain.
        # This case (all layers empty) might need specific handling if start/end should be None or raise error.
        # Current TimeSpan default is start=0.0, end=0.0.
        # If all layers in a hierarchy are empty segmentations, they'll have start=0, end=0.
        # So hier_start/end would be 0.

        # Final validation for TimeSpan start/end
        if self.start > self.end:
            raise ValueError(f"Derived start time ({self.start}) must be ≤ derived end time ({self.end}).")


    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx] # layers is now a tuple

    @property
    def intervals(self) -> Tuple[np.ndarray, ...]: # Return tuple for immutability
        return tuple(layer.intervals for layer in self.layers)

    @property
    def labels(self) -> Tuple[Tuple[str | None, ...], ...]: # Return tuple of tuples
        return tuple(layer.labels for layer in self.layers) # layer.labels is already tuple

    @property
    def boundaries(self) -> Tuple[Tuple[float, ...], ...]: # Return tuple of tuples
        return tuple(layer.boundaries for layer in self.layers) # layer.boundaries is already tuple

    def __repr__(self) -> str:
        name_str = f"name='{self.name}'" if self.name is not None else "name='None'"
        return f"Hierarchy({name_str}, {len(self)} layers, duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        return f"Hierarchy(name='{self.name}', {len(self)} layers, duration={self.duration:.2f}s)"

    def plot(self, ax: Axes | None = None, text: bool = True, **kwargs: Any) -> tuple[Figure | SubFigure, Axes]:
        from .viz import label_style_dict # type: ignore

        figsize = kwargs.pop("figsize", None)
        n_layers = len(self.layers)
        # Ensure n_layers is at least 1 for subplots, __post_init__ guarantees this.
        fig, axes_obj = plt.subplots( # type: ignore
            n_layers, 1, figsize=figsize or (6, 0.5 + 0.5 * n_layers), sharex=True, constrained_layout=True
        )

        # Ensure axes_obj is always a list-like structure (array of Axes)
        axes_list: List[Axes] = []
        if n_layers == 1:
            axes_list = [axes_obj] # type: ignore
        elif n_layers > 1:
            axes_list = list(axes_obj) # type: ignore

        for i, (layer, current_ax) in enumerate(zip(self.layers, axes_list)):
            layer_labels = layer.labels # Should be Tuple[str|None,...]
            layer.plot(
                ax=current_ax,
                style_map=label_style_dict(layer_labels), # type: ignore
                title=False,
                ytick=f"Level {i}",
                time_ticks=(i == n_layers - 1),
            )

        # Since n_layers >= 1 is guaranteed by __post_init__
        axes_list[-1].set_xlabel("Time (s)")
        return fig, axes_list[-1]


    def plot_single_axis(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: bool = True,
        time_ticks: bool = True,
        layer_height: float = 0.8,
        layer_gap: float = 0.1,
    ) -> tuple[Figure | SubFigure, Axes]:
        from .viz import label_style_dict # type: ignore

        n_layers = len(self.layers) # Guaranteed >= 1
        fig: Figure | SubFigure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (10, 2 + 0.5 * n_layers)) # type: ignore
        else:
            fig = ax.figure

        total_height = n_layers * (layer_height + layer_gap) - layer_gap
        current_y = total_height

        for layer_idx, layer in enumerate(self.layers): # Use enumerate for index
            layer_ymin = current_y - layer_height
            self._plot_layer_on_axis(ax, layer, label_style_dict, layer_ymin, layer_height, total_height)
            current_y -= layer_height + layer_gap

        if title and self.name:
            ax.set_title(self.name)

        xlim_start = self.start
        xlim_end = self.end
        ax.set_xlim(xlim_start, xlim_end) if xlim_start != xlim_end else ax.set_xlim(-0.1, 0.1) # type: ignore

        ax.set_ylim(0, total_height)

        if time_ticks:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # type: ignore
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticks([])

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
        layer_labels = layer.labels # tuple of labels
        style_map = style_provider(layer_labels)
        layer_ymax = layer_ymin + layer_height

        for span in layer.segments: # segments is tuple of TimeSpans
            span_style = style_map.get(span.name or "", {})

            ymin_norm = layer_ymin / total_height if total_height > 0 else 0
            ymax_norm = layer_ymax / total_height if total_height > 0 else 0

            ax.axvspan(
                span.start,
                span.end,
                ymin=ymin_norm,
                ymax=ymax_norm,
                alpha=span_style.pop("alpha", 0.7), # Get alpha, remove from dict
                **span_style, # Pass remaining style attributes
            )

            if span.name:
                current_start = self.start
                current_end = self.end
                text_offset_val = 0.005 * max(current_end - current_start, 0.1) # type: ignore

                ax.text(
                    span.start + text_offset_val,
                    layer_ymin + layer_height / 2,
                    span.name,
                    va="center",
                    ha="left",
                    fontsize=7,
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
                )

    @classmethod
    def from_jams(cls, jams_annotation: "jams.Annotation", name: str | None = None) -> "Hierarchy": # type: ignore
        if jams_annotation.namespace != "multi_segment": # type: ignore
            raise ValueError(f"Expected 'multi_segment' namespace, got '{jams_annotation.namespace}'") # type: ignore

        from jams.eval import hierarchy_flatten # type: ignore

        hier_intervals, hier_labels = hierarchy_flatten(jams_annotation) # type: ignore

        segmentations = []
        for i, (intervals, labels) in enumerate(zip(hier_intervals, hier_labels)):
            current_intervals_np = np.array(intervals) if not isinstance(intervals, np.ndarray) else intervals
            current_labels_list = list(labels) if not isinstance(labels, list) else labels
            # Segmentation.from_intervals expects name as keyword argument
            seg = Segmentation.from_intervals(intervals=current_intervals_np, labels=current_labels_list, name=f"level_{i}")
            segmentations.append(seg)

        final_name = name
        if not final_name:
            annotator_meta = jams_annotation.annotation_metadata.annotator # type: ignore
            if annotator_meta and "name" in annotator_meta: # type: ignore
                final_name = annotator_meta["name"] # type: ignore

        return cls(name=final_name, layers=segmentations)

    @classmethod
    def from_json(cls, json_data: List[List[List[Any]]], name: str | None = None) -> "Hierarchy":
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

            segments_for_layer = [] # Corrected variable name
            for interval, label in zip(intervals_data, labels_data):
                actual_interval = (
                    interval[0]
                    if (isinstance(interval, list) and len(interval) == 1 and isinstance(interval[0], list))
                    else interval
                )

                if not (isinstance(actual_interval, list) and len(actual_interval) == 2):
                    raise ValueError(f"Layer {i}: malformed interval {actual_interval}.")

                try:
                    start_time, end_time = float(actual_interval[0]), float(actual_interval[1])
                    segments_for_layer.append(TimeSpan(start=start_time, end=end_time, name=str(label)))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Layer {i}: invalid interval {actual_interval}: {e}") from e

            segmentations.append(Segmentation(name=f"layer_{i}", segments=segments_for_layer))

        return cls(name=name, layers=segmentations)


@dataclass(frozen=True)
class RatedBoundaries:
    """
    A canonical representation of boundaries as a flat list of (timestamp, salience)
    pairs. This is the central object of the "Feature World" and supports a
    fluent interface for progressive refinement.

    Events are stored as a tuple of (timestamp, salience) tuples.
    """
    events: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Basic validation:
        for event_idx, event in enumerate(self.events): # Use enumerate for index access if needed
            if not (isinstance(event, tuple) and len(event) == 2):
                raise ValueError(
                    f"Event at index {event_idx} is not a tuple of length 2: {event}"
                )
            if not (isinstance(event[0], (float, int)) and isinstance(event[1], (float, int))):
                raise ValueError(
                    f"Event {event} at index {event_idx}: timestamp and salience must be numeric."
                )
            if event[0] < 0:
                raise ValueError(f"Event {event} at index {event_idx}: timestamp ({event[0]}) cannot be negative.")

        # Check for sorted order by timestamp.
        # If not sorted, it's often better to raise an error for a "canonical" representation
        # or clearly document that sorting is the responsibility of the creator.
        for i in range(len(self.events) - 1):
            if self.events[i][0] > self.events[i+1][0]:
                raise ValueError(
                    f"Events in RatedBoundaries must be sorted by timestamp. "
                    f"Event at index {i} ({self.events[i][0]}) > event at index {i+1} ({self.events[i+1][0]})."
                )

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        return self.events[idx]

    def __iter__(self):
        return iter(self.events)


@dataclass(frozen=True)
class ProperHierarchy(Hierarchy):
    """
    A guaranteed-monotonic hierarchy, derived from a set of rated boundaries.
    It "is-a" Hierarchy and can be used by any function that expects one, but
    its internal layer structure is synthesized from its canonical boundary list.
    This class does not add new public fields beyond what Hierarchy provides.
    It is constructed differently, primarily via from_rated_boundaries.
    """

    @classmethod
    def from_rated_boundaries(
        cls,
        rated_boundaries: RatedBoundaries,
        name: str | None = None,
    ) -> "ProperHierarchy":
        """
        The primary constructor. Synthesizes a ProperHierarchy from a
        quantized list of rated boundaries.
        """
        processed_events: List[Tuple[float, int]] = []
        for ts, salience in rated_boundaries.events:
            if not (isinstance(salience, int) or (isinstance(salience, float) and salience.is_integer())):
                raise ValueError(
                    f"Salience value '{salience}' for timestamp {ts} cannot be cast to an integer depth."
                )
            if salience < 0:
                raise ValueError(
                    f"Salience value '{salience}' for timestamp {ts} must be a non-negative integer depth."
                )
            processed_events.append((ts, int(salience)))

        layers = cls._build_layers_from_rated(tuple(processed_events))

        if not layers:
            default_timespan = TimeSpan(start=0.0, end=0.0, name="default_segment") # Ensure name is str or None
            default_segmentation = Segmentation(name="level_0", segments=[default_timespan])
            layers = [default_segmentation]

        return cls(name=name, layers=layers)

    @staticmethod
    def _build_layers_from_rated(events: Tuple[Tuple[float, int], ...]) -> List[Segmentation]:
        """
        Constructs a list of Segmentation objects from rated boundaries.
        A boundary (t, d) signifies that timestamp 't' is a structural boundary
        at hierarchical depth 'd'.
        Rule: Boundaries of layer 'L_idx' include all (t,d_event) where d_event <= L_idx.
        (Assuming depth d_event means it's significant *up to* level d_event,
        and layer indices L_idx also run from 0=coarsest).
        """
        if not events:
            return []

        max_event_depth = -1
        for _, depth_val in events:
            if depth_val > max_event_depth:
                max_event_depth = depth_val

        if max_event_depth < 0:
            return []

        num_layers = max_event_depth + 1
        all_generated_layers: List[Segmentation] = []

        all_event_timestamps = sorted(list(set(ts for ts, _ in events)))

        global_start_time = all_event_timestamps[0] if all_event_timestamps else 0.0
        global_end_time = all_event_timestamps[-1] if all_event_timestamps else 0.0

        for current_layer_idx in range(num_layers):
            # Collect boundaries for this layer: all (t,d_event) where d_event <= current_layer_idx
            layer_boundaries_set: set[float] = set()
            for ts, event_depth in events:
                if event_depth <= current_layer_idx:
                    layer_boundaries_set.add(ts)

            current_layer_boundaries = sorted(list(layer_boundaries_set))

            # Ensure the layer spans the global start and end times.
            final_segmentation_boundaries: List[float]
            if not current_layer_boundaries:
                # This layer has no specific boundaries from events (e.g., current_layer_idx is too coarse).
                # It should still exist as a single segment spanning the global duration.
                if global_start_time == global_end_time:
                    final_segmentation_boundaries = [global_start_time]
                else:
                    final_segmentation_boundaries = [global_start_time, global_end_time]
            else:
                # Add global start/end if not encompassed by this layer's event boundaries.
                if current_layer_boundaries[0] > global_start_time:
                     current_layer_boundaries.insert(0, global_start_time)
                if current_layer_boundaries[-1] < global_end_time:
                     current_layer_boundaries.append(global_end_time)
                final_segmentation_boundaries = sorted(list(set(current_layer_boundaries)))

            # If after all this, final_segmentation_boundaries is empty (e.g. global_start/end were not well-defined, though handled above)
            # or results in a situation Segmentation.from_boundaries can't handle, we might need a fallback.
            # However, with global_start/end, it should always have at least one point.
            if not final_segmentation_boundaries: # Should ideally not be reached if global times are set
                final_segmentation_boundaries = [0.0] # Default to a single point at 0.0

            # Segments are unnamed by default in this construction.
            num_segments = max(0, len(final_segmentation_boundaries) -1) if len(final_segmentation_boundaries) > 1 else (1 if len(final_segmentation_boundaries) == 1 else 0)

            segment_labels: List[str|None] = [None] * num_segments

            layer_segmentation = Segmentation.from_boundaries(
                boundaries=final_segmentation_boundaries,
                labels=segment_labels,
                name=f"level_{current_layer_idx}"
            )
            all_generated_layers.append(layer_segmentation)

        return all_generated_layers
