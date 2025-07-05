"""
This module provides the main pipeline for synthesizing monotonic hierarchies.
It uses a strategy-based approach to allow for flexible and extensible
transformations from a generic Hierarchy to a ProperHierarchy.
"""

from __future__ import annotations

import dataclasses

from .core import Hierarchy, ProperHierarchy
from .strategies import (
    BoundaryGroupingStrategy,
    LevelGroupingStrategy,
    SalienceStrategy,
)


class Pipeline:
    """
    Orchestrates the end-to-end process of transforming a raw `Hierarchy`
    into a `ProperHierarchy` using a sequence of pluggable strategies.

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
            The strategy to calculate a salience value for each boundary.
        grouping_strategy : BoundaryGroupingStrategy | None
            The strategy to consolidate boundaries that are close in time. If
            None, this step is skipped.
        leveling_strategy : LevelGroupingStrategy
            The strategy to quantize salience values into discrete levels and
            synthesize the final `ProperHierarchy`.
        """
        self.salience_strategy = salience_strategy
        self.grouping_strategy = grouping_strategy
        self.leveling_strategy = leveling_strategy

    def process(self, hierarchy: Hierarchy, name: str | None = None) -> ProperHierarchy:
        """
        Executes the full transformation pipeline on a given hierarchy.

        The process follows these steps:
        1.  **Salience Calculation:** A `SalienceStrategy` is used to compute
            a `RatedBoundaries` object from the input `Hierarchy`.
        2.  **Boundary Grouping (Optional):** If a `BoundaryGroupingStrategy`
            is provided, it consolidates nearby boundaries.
        3.  **Level Quantization & Synthesis:** A `LevelGroupingStrategy`
            converts the rated boundaries into a `ProperHierarchy`.

        Parameters
        ----------
        hierarchy : Hierarchy
            The input hierarchy to process.
        name : str | None, optional
            An optional name for the resulting `ProperHierarchy`. If None, the
            name of the input hierarchy is used.

        Returns
        -------
        ProperHierarchy
            A new, monotonically nested hierarchy.
        """
        # 1. Analyze salience
        rated_boundaries = self.salience_strategy.calculate(hierarchy)

        # 2. Group boundaries in time (optional)
        if self.grouping_strategy:
            rated_boundaries = rated_boundaries.group_boundaries(self.grouping_strategy)

        # 3. Quantize saliences into levels and synthesize
        proper_hierarchy = rated_boundaries.quantize_level(self.leveling_strategy)

        # 4. Return the final object with the correct name
        final_name = name if name is not None else hierarchy.name
        return dataclasses.replace(proper_hierarchy, name=final_name)
