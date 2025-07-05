"""Pluggable strategy interfaces for monotonic casting experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import (
        MultiSegment,
        Hierarchy, # ProperHierarchy is the new Hierarchy
        RatedBoundaries,
        RatedBoundary,
        LeveledBoundary,
        Boundary,
        Segmentation,
    )


class SalienceStrategy(ABC): # Conceptually, this is now a "RateStrategy"
    """
    Abstract base class for strategies that calculate boundary rate/salience.

    A rate strategy is responsible for the first step in the synthesis
    pipeline: converting a generic `MultiSegment` into `RatedBoundaries` by
    assigning a numerical rate (formerly salience) value to each unique boundary time.
    """

    @abstractmethod
    def calculate(self, multi_segment: MultiSegment) -> RatedBoundaries:
        """
        Takes a MultiSegment and returns a RatedBoundaries object.

        Parameters
        ----------
        multi_segment : MultiSegment
            The input multi-segment structure to analyze.

        Returns
        -------
        RatedBoundaries
            A new collection of boundaries, each with an assigned rate.
        """
        pass


class BoundaryGroupingStrategy(ABC):
    """
    Abstract base class for strategies that group or consolidate boundaries.

    This strategy operates on a sequence of `RatedBoundary` objects, typically
    to merge events that are very close in time, which can help clean up
    artifacts from upstream analyses.
    """

    @abstractmethod
    def group(self, boundaries: Sequence[RatedBoundary]) -> Sequence[RatedBoundary]:
        """
        Takes a sequence of RatedBoundary objects and returns a consolidated version.

        Parameters
        ----------
        boundaries : Sequence[RatedBoundary]
            The input sequence of rated boundaries.

        Returns
        -------
        Sequence[RatedBoundary]
            A new, potentially smaller, sequence of consolidated boundaries.
        """
        pass


class LevelGroupingStrategy(ABC):
    """
    Abstract base class for strategies that quantize saliences and synthesize a hierarchy.

    This is the final stage of the pipeline, responsible for converting a
    `RatedBoundaries` object (with continuous or discrete rates) into a
    `Hierarchy` with monotonically nested layers.
    The synthesized `Hierarchy` must contain `LeveledBoundary` instances.
    """

    @abstractmethod
    def quantize(self, boundaries: RatedBoundaries) -> Hierarchy:
        """
        Takes a RatedBoundaries object, quantizes its rates into discrete
        levels, and synthesizes a final `Hierarchy`.
        The synthesized Hierarchy's layers must contain `LeveledBoundary` instances.

        Parameters
        ----------
        boundaries : RatedBoundaries
            The input collection of rated boundaries.

        Returns
        -------
        Hierarchy
            The final, synthesized monotonic hierarchy.
        """
        pass


# --- Concrete Implementations of the Contracts ---


class DirectSynthesisStrategy(LevelGroupingStrategy):
    """
    Synthesizes a `Hierarchy` from rated boundaries based on rate thresholds.

        This strategy creates one layer for each unique rate value found in the
        boundaries. A layer contains all boundaries with a rate greater than
        or equal to that layer's threshold, resulting in a monotonically nested
        hierarchy. Assumes higher rate values are more structurally important.
        Boundaries in the output Hierarchy are `LeveledBoundary` instances, where
        the 'level' is derived from the 'rate'.
    """

    def quantize(self, boundaries: RatedBoundaries) -> Hierarchy:
        from .core import LeveledBoundary, Hierarchy, Segmentation, Boundary

        events = boundaries.events
        start_time, end_time = boundaries.start_time, boundaries.end_time
        duration = end_time - start_time

        # Ensure all rates are positive integers for LeveledBoundary compatibility
        # This might involve rounding or other logic if rates can be floats.
        # For now, we assume rates are convertible to positive integers.
        # A more robust implementation might handle non-integer rates gracefully.
        processed_events: list[RatedBoundary] = []
        for event in events:
            if not isinstance(event.rate, int) or event.rate <= 0:
                # This is a simple way to handle it; could be more sophisticated
                # e.g., by scaling, flooring, or raising an error.
                # For this refactor, we assume rates are usable as levels.
                # If rates are float, they need conversion to int.
                # If rates are not positive, they violate LeveledBoundary constraints.
                # This part of the logic is crucial for the new Hierarchy requirements.
                # For now, let's assume event.rate is already a positive integer.
                # If not, this strategy needs adjustment or the input RatedBoundary rates need prior processing.
                if isinstance(event.rate, float): # Example: convert float rate to int level
                    level_val = int(round(event.rate))
                    if level_val <=0: level_val = 1 # Ensure positive
                elif isinstance(event.rate, int) and event.rate > 0:
                    level_val = event.rate
                else:
                    # Skip or handle invalid rate for LeveledBoundary
                    # For simplicity, we'll skip if it's not a positive integer after potential conversion
                    print(f"Warning: Skipping event with non-positive integer rate {event.rate} for LeveledBoundary.")
                    continue
                processed_events.append(RatedBoundary(event.time, level_val, event.label))
            else: # Rate is already a positive integer
                 processed_events.append(event)

        events = processed_events


        if not events:
            # Need to use LeveledBoundary for start/end if creating an empty Hierarchy
            # However, Hierarchy requires layers, and layers require boundaries.
            # An empty Hierarchy should probably still have start/end LeveledBoundaries if it's truly empty of events.
            # The __post_init__ of Hierarchy will complain if boundaries are not LeveledBoundary.
            # For an empty event set, we might return a Hierarchy with one layer spanning start to end,
            # using a default level (e.g., 1) for its boundaries.
            # Or, if Hierarchy can be empty (no layers), then this is simpler.
            # Based on MultiSegment, it needs at least one layer.
            # Let's create a single layer with start/end boundaries if no events.
            b_start = LeveledBoundary(start_time, level=1) # Default level 1
            b_end = LeveledBoundary(end_time, level=1)     # Default level 1
            single_layer = Segmentation(start=b_start, duration=duration, boundaries=[b_start, b_end], label="empty_level")
            return Hierarchy(start=b_start, duration=duration, layers=[single_layer], label="direct_synthesis_empty")


        # Get unique rate levels (which must be positive integers), reverse from coarse (high) to fine (low)
        unique_rate_levels = sorted(list(set(int(e.rate) for e in events if isinstance(e.rate, int) and e.rate > 0)), reverse=True)
        if not unique_rate_levels and events: # If all rates were invalid
             b_start = LeveledBoundary(start_time, level=1)
             b_end = LeveledBoundary(end_time, level=1)
             single_layer = Segmentation(start=b_start, duration=duration, boundaries=[b_start, b_end], label="default_level")
             return Hierarchy(start=b_start, duration=duration, layers=[single_layer], label="direct_synthesis_default")

        all_layers = []

        for i, rate_thresh in enumerate(unique_rate_levels):
            # A layer contains all boundaries with rate >= the current threshold
            # These must be LeveledBoundary instances.
            layer_boundaries_times_and_labels: dict[float, str | None] = {}
            for b_event in events:
                if isinstance(b_event.rate, int) and b_event.rate >= rate_thresh:
                    # Keep the label from the original RatedBoundary
                    if b_event.time not in layer_boundaries_times_and_labels:
                         layer_boundaries_times_and_labels[b_event.time] = b_event.label

            # Add start and end points for the segmentation layer
            # Their labels will be None unless derived from events at these exact times.
            if start_time not in layer_boundaries_times_and_labels:
                layer_boundaries_times_and_labels[start_time] = None
            if end_time not in layer_boundaries_times_and_labels:
                layer_boundaries_times_and_labels[end_time] = None

            # Create LeveledBoundary objects for this layer
            # The level for these boundaries is the current rate_thresh, as all included boundaries meet this.
            # However, the problem states LeveledBoundary.level is a refinement of RatedBoundary.rate.
            # The individual boundary's own rate (which is now its level) should be used.
            # The layer is defined by rate_thresh, but boundaries within it retain their specific levels.

            current_layer_leveled_boundaries: list[LeveledBoundary] = []
            for t_event in events: # Iterate through original events to get their specific rates for levels
                if isinstance(t_event.rate, int) and t_event.rate >= rate_thresh :
                    # Ensure we don't add duplicate times if multiple events had the same time but different original labels/rates
                    # For simplicity, we'll use the properties of the first event matching the time.
                    # A more robust approach might merge/select labels/rates.
                    if not any(lb.time == t_event.time for lb in current_layer_leveled_boundaries):
                        current_layer_leveled_boundaries.append(LeveledBoundary(time=t_event.time, level=int(t_event.rate), label=t_event.label))

            # Add start and end LeveledBoundaries if not already present from events
            if not any(lb.time == start_time for lb in current_layer_leveled_boundaries):
                 current_layer_leveled_boundaries.append(LeveledBoundary(time=start_time, level=rate_thresh, label=layer_boundaries_times_and_labels.get(start_time))) # Use rate_thresh as level for start/end
            if not any(lb.time == end_time for lb in current_layer_leveled_boundaries):
                 current_layer_leveled_boundaries.append(LeveledBoundary(time=end_time, level=rate_thresh, label=layer_boundaries_times_and_labels.get(end_time)))

            sorted_b_objs = sorted(list(current_layer_leveled_boundaries))

            layer_label = f"level_rate_{rate_thresh}"
            # The start boundary for the Segmentation should be the first LeveledBoundary in sorted_b_objs
            seg_start_boundary = sorted_b_objs[0]

            all_layers.append(
                Segmentation(start=seg_start_boundary, duration=duration, boundaries=sorted_b_objs, label=layer_label)
            )

        if not all_layers: # Should not happen if unique_rate_levels was populated
            b_start = LeveledBoundary(start_time, level=1)
            b_end = LeveledBoundary(end_time, level=1)
            single_layer = Segmentation(start=b_start, duration=duration, boundaries=[b_start, b_end], label="fallback_level")
            all_layers.append(single_layer)

        # The hierarchy's overall start/duration should match the input boundaries.
        # The start boundary for the Hierarchy itself should also be a LeveledBoundary.
        # We can derive its level from the coarsest layer (first layer in all_layers).
        hierarchy_start_boundary = all_layers[0].boundaries[0] # This is already a LeveledBoundary
        if not isinstance(hierarchy_start_boundary, LeveledBoundary): # Should be, but as a safeguard
            hierarchy_start_boundary = LeveledBoundary(hierarchy_start_boundary.time, level=1, label=hierarchy_start_boundary.label)


        return Hierarchy( # type: ignore
            start=hierarchy_start_boundary, # Must be LeveledBoundary
            duration=duration,
            layers=all_layers,
            label="direct_synthesis",
        )


class FrequencyStrategy(SalienceStrategy):
    """
    Calculates rate (formerly salience) based on the frequency of a boundary's appearance.

    The rate of each boundary is the number of layers in the input
    `MultiSegment` that it appears in. A boundary present in more layers is
    considered to have a higher rate.
    """

    def calculate(self, multi_segment: MultiSegment) -> RatedBoundaries:
        from .core import RatedBoundaries, RatedBoundary

        boundary_counts: dict[float, int] = {}
        for layer in multi_segment.layers:
            for b in layer.boundaries:
                boundary_counts[b.time] = boundary_counts.get(b.time, 0) + 1

        # Create RatedBoundary objects
        # The 'rate' here is the count, which is an integer.
        # This is compatible with LeveledBoundary if used directly, assuming count > 0.
        # Pipeline might need to ensure counts are positive if this output is directly used by a LeveledBoundary constructor.
        rated_boundaries = [RatedBoundary(time=time, rate=count, label=None) for time, count in boundary_counts.items() if count > 0] # Ensure rate > 0 for LeveledBoundary
        return RatedBoundaries(
            events=sorted(rated_boundaries),
            start_time=multi_segment.start.time,
            end_time=multi_segment.end.time,
        )


class CoarsestNonzeroStrategy(SalienceStrategy):
    """
    Calculates rate (formerly salience) based on the number of layers a boundary is a member of.

    A boundary's rate is the total number of layers minus the index of the
    coarsest layer it appears in. For a multi-segment structure with N layers, a boundary
    first appearing in layer `i` (0-indexed, coarse to fine) gets a rate
    of `N - i`. This means boundaries in coarser layers get higher scores.
    """

    def calculate(self, multi_segment: MultiSegment) -> RatedBoundaries:
        from .core import RatedBoundaries, RatedBoundary

        num_layers = len(multi_segment.layers)
        boundary_rates: dict[float, int] = {} # Changed from boundary_salience
        for i, layer in enumerate(multi_segment.layers):
            for b in layer.boundaries:
                if b.time not in boundary_rates:
                    rate_val = num_layers - i
                    if rate_val > 0 : # Ensure rate is positive for LeveledBoundary compatibility
                        boundary_rates[b.time] = rate_val

        # Create RatedBoundary objects. Rates are positive integers here.
        rated_boundaries = [RatedBoundary(time=time, rate=rate, label=None) for time, rate in boundary_rates.items()]
        return RatedBoundaries(
            events=sorted(rated_boundaries),
            start_time=multi_segment.start.time,
            end_time=multi_segment.end.time,
        )
