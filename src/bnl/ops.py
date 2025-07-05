"""
This module provides the main pipeline for synthesizing monotonic hierarchies.
It uses a strategy-based approach to allow for flexible and extensible
transformations from a generic Hierarchy to a ProperHierarchy.
"""

from __future__ import annotations

import dataclasses

from .core import MultiSegment, Hierarchy
from .strategies import (
    BoundaryGroupingStrategy,
    LevelGroupingStrategy,
    SalienceStrategy,
)


class Pipeline:
    """
    Orchestrates the end-to-end process of transforming a raw `MultiSegment`
    into a `Hierarchy` using a sequence of pluggable strategies.

    This class provides a high-level API to convert potentially non-monotonic
    or unstructured boundary data into a formal, monotonically nested hierarchy.
    """

    def __init__(
        self,
        salience_strategy: SalienceStrategy,
        grouping_strategy: BoundaryGroupingStrategy | None,
        leveling_strategy: LevelGroupingStrategy,
    ):
        """
        Initializes the pipeline with a set of strategies.

        Parameters
        ----------
        salience_strategy : SalienceStrategy
            The strategy to calculate a rate value for each boundary.
        grouping_strategy : BoundaryGroupingStrategy | None
            The strategy to consolidate boundaries that are close in time. If
            None, this step is skipped.
        leveling_strategy : LevelGroupingStrategy
            The strategy to quantize rate values into discrete levels and
            synthesize the final `Hierarchy`.
        """
        self.salience_strategy = salience_strategy
        self.grouping_strategy = grouping_strategy
        self.leveling_strategy = leveling_strategy

    def __call__(self, multi_segment: MultiSegment, label: str | None = None) -> Hierarchy:
        """
        Executes the full transformation pipeline on a given multi-segment structure.

        The process follows these steps:
        1.  **Rate Calculation:** A `SalienceStrategy` (conceptually now a "RateStrategy")
            is used to compute a `RatedBoundaries` object from the input `MultiSegment`.
        2.  **Boundary Grouping (Optional):** If a `BoundaryGroupingStrategy`
            is provided, it consolidates nearby boundaries.
        3.  **Level Quantization & Synthesis:** A `LevelGroupingStrategy`
            converts the rated boundaries into a `Hierarchy`.

        Parameters
        ----------
        multi_segment : MultiSegment
            The input multi-segment structure to process.
        label : str | None, optional
            An optional label for the resulting `Hierarchy`. If None, the
            label of the input multi-segment structure is used.

        Returns
        -------
        Hierarchy
            A new, monotonically nested hierarchy.
        """
        # 1. Analyze salience/rate
        rated_boundaries = self.salience_strategy.calculate(multi_segment)

        # 2. Group boundaries in time (optional)
        if self.grouping_strategy:
            rated_boundaries = rated_boundaries.group_boundaries(self.grouping_strategy)

        # 3. Quantize saliences into levels and synthesize
        hierarchy = rated_boundaries.quantize_level(self.leveling_strategy)

        # 4. Return the final object with the correct label
        final_label = label if label is not None else multi_segment.label
        return dataclasses.replace(hierarchy, label=final_label)
