"""Pluggable strategy interfaces for monotonic casting experiments."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Hierarchy, ProperHierarchy, RatedBoundaries, RatedBoundary


class SalienceStrategy(ABC):
    """
    Abstract base class for strategies that calculate boundary salience.

    A salience strategy is responsible for the first step in the synthesis
    pipeline: converting a generic `Hierarchy` into `RatedBoundaries` by
    assigning a numerical salience value to each unique boundary time.
    """

    @abstractmethod
    def calculate(self, hierarchy: Hierarchy) -> RatedBoundaries:
        """
        Takes a Hierarchy and returns a RatedBoundaries object.

        Parameters
        ----------
        hierarchy : Hierarchy
            The input hierarchy to analyze.

        Returns
        -------
        RatedBoundaries
            A new collection of boundaries, each with an assigned salience.
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
    `RatedBoundaries` object (with continuous or discrete saliences) into a
    `ProperHierarchy` with monotonically nested layers.
    """

    @abstractmethod
    def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
        """
        Takes a RatedBoundaries object, quantizes its saliences into discrete
        levels, and synthesizes a final ProperHierarchy.

        Parameters
        ----------
        boundaries : RatedBoundaries
            The input collection of rated boundaries.

        Returns
        -------
        ProperHierarchy
            The final, synthesized monotonic hierarchy.
        """
        pass


# --- Concrete Implementations of the Contracts ---


class DirectSynthesisStrategy(LevelGroupingStrategy):
    """
    Synthesizes a `ProperHierarchy` directly from rated boundaries.

    This strategy assumes that the salience values in the `RatedBoundaries`
    are already integer-like and represent discrete hierarchy levels (e.g.,
    0, 1, 2...). It creates one layer for each integer level from 0 up to the
    maximum salience value found.
    """

    def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
        from .core import Boundary, ProperHierarchy, Segmentation

        rated_boundaries = boundaries.events
        if not rated_boundaries:
            return ProperHierarchy(layers=[Segmentation(boundaries=[Boundary(boundaries.start_time)], labels=[])])

        # Sort by time, then by salience for stable ordering
        events = tuple(sorted(rated_boundaries, key=lambda e: (e.time, e.salience)))

        # Convert saliences to integer depths
        int_events = []
        for event in events:
            if not (isinstance(event.salience, int) or event.salience.is_integer()):
                raise ValueError(f"Salience {event.salience} for boundary at {event.time} must be an integer.")
            int_events.append(dataclasses.replace(event, salience=int(event.salience)))

        max_depth = max((e.salience for e in int_events), default=-1)
        if max_depth < 0:
            return ProperHierarchy(layers=[Segmentation(boundaries=[Boundary(boundaries.start_time)], labels=[])])

        num_layers = int(max_depth) + 1
        all_layers = []

        for depth_level in range(num_layers):
            # Boundaries for this level are all events with salience <= current depth
            layer_boundary_times = {boundaries.start_time, boundaries.end_time}
            for event in int_events:
                if event.salience <= depth_level:
                    layer_boundary_times.add(event.time)

            sorted_times = sorted(list(layer_boundary_times))
            b_objs = [Boundary(t) for t in sorted_times]
            labels = [None] * (len(b_objs) - 1)
            layer_name = f"level_{depth_level}"
            all_layers.append(Segmentation(boundaries=b_objs, labels=labels, name=layer_name))

        return ProperHierarchy(layers=all_layers)


class FrequencyStrategy(SalienceStrategy):
    """
    Calculates salience based on the frequency of a boundary's appearance.

    The salience of each boundary is the number of layers in the input
    `Hierarchy` that it appears in. A boundary present in more layers is
    considered more salient.
    """

    def calculate(self, hierarchy: Hierarchy) -> RatedBoundaries:
        from .core import RatedBoundaries, RatedBoundary

        boundary_counts: dict[float, int] = {}
        for layer in hierarchy.layers:
            for b in layer.boundaries:
                boundary_counts[b.time] = boundary_counts.get(b.time, 0) + 1

        # Create RatedBoundary objects
        rated_boundaries = [RatedBoundary(time=time, salience=count) for time, count in boundary_counts.items()]
        return RatedBoundaries(
            events=sorted(rated_boundaries),
            start_time=hierarchy.start.time,
            end_time=hierarchy.end.time,
        )


class CoarsestNonzeroStrategy(SalienceStrategy):
    """
    Calculates salience based on the coarsest layer a boundary appears in.

    The salience is the index of the *first* (coarsest) layer in which the
    boundary is present. A boundary appearing in layer 0 will have a salience
    of 0, a new boundary appearing first in layer 1 will have salience 1, etc.
    This assumes layers are ordered from coarse to fine.
    """

    def calculate(self, hierarchy: Hierarchy) -> RatedBoundaries:
        from .core import RatedBoundaries, RatedBoundary

        boundary_salience: dict[float, int] = {}
        for i, layer in enumerate(hierarchy.layers):
            for b in layer.boundaries:
                if b.time not in boundary_salience:
                    boundary_salience[b.time] = i

        rated_boundaries = [RatedBoundary(time=time, salience=salience) for time, salience in boundary_salience.items()]
        return RatedBoundaries(
            sorted(rated_boundaries),
            start_time=hierarchy.start.time,
            end_time=hierarchy.end.time,
        )
