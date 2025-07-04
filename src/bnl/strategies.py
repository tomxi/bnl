"""Pluggable strategy interfaces for monotonic casting experiments."""

from __future__ import annotations

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
    Synthesizes a `ProperHierarchy` from rated boundaries based on salience thresholds.

        This strategy creates one layer for each unique salience value found in the
        boundaries. A layer contains all boundaries with a salience greater than
        or equal to that layer's threshold, resulting in a monotonically nested
        hierarchy. Assumes higher salience values are more structurally important.
    """

    def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
        from .core import Boundary, ProperHierarchy, Segmentation

        events = boundaries.events
        start_time, end_time = boundaries.start_time, boundaries.end_time
        duration = end_time - start_time

        if not events:
            return ProperHierarchy(start=Boundary(start_time), duration=duration, layers=[], label="direct_synthesis")

        # Get unique salience levels, but reverse from coarse (high) to fine (low)
        unique_salience_levels = sorted(list(set(e.salience for e in events)), reverse=True)
        all_layers = []

        for i, salience_thresh in enumerate(unique_salience_levels):
            # A layer contains all boundaries with salience >= the current threshold
            level_boundaries = {b.time for b in events if b.salience >= salience_thresh}
            level_boundaries.add(start_time)
            level_boundaries.add(end_time)

            b_objs = [Boundary(t) for t in sorted(list(level_boundaries))]
            layer_label = f"level_{i}"

            all_layers.append(
                Segmentation(start=Boundary(start_time), duration=duration, boundaries=b_objs, label=layer_label)
            )

        # The hierarchy's overall start/duration should match the input boundaries
        return ProperHierarchy(
            start=Boundary(start_time),
            duration=duration,
            layers=all_layers,
            label="direct_synthesis",
        )


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
    Calculates salience based on the number of layers a boundary is a member of.

    A boundary's salience is the total number of layers minus the index of the
    coarsest layer it appears in. For a hierarchy with N layers, a boundary
    first appearing in layer `i` (0-indexed, coarse to fine) gets a salience
    of `N - i`. This means boundaries in coarser layers get higher scores.
    """

    def calculate(self, hierarchy: Hierarchy) -> RatedBoundaries:
        from .core import RatedBoundaries, RatedBoundary

        num_layers = len(hierarchy.layers)
        boundary_salience: dict[float, int] = {}
        for i, layer in enumerate(hierarchy.layers):
            for b in layer.boundaries:
                if b.time not in boundary_salience:
                    boundary_salience[b.time] = num_layers - i

        rated_boundaries = [RatedBoundary(time=time, salience=salience) for time, salience in boundary_salience.items()]
        return RatedBoundaries(
            sorted(rated_boundaries),
            start_time=hierarchy.start.time,
            end_time=hierarchy.end.time,
        )
