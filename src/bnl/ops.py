"""
This module provides the core algorithmic operations for transforming
boundary and hierarchy objects.

The functions in this module are designed to be composed into pipelines,
either directly or through the fluent API provided by the `bnl.core` classes.
"""

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

from .core import RatedBoundary

if TYPE_CHECKING:
    from .core import BoundaryContour, MultiSegment

__all__ = [
    "LabelContextMap",
    "SaliencePayload",
    "naive_salience",
]

# Type alias for the label context map
LabelContextMap = dict[float, list[str]]


class SaliencePayload(NamedTuple):
    """
    A payload containing both boundary contour and label context.

    This is the intermediate result from analysis functions that contains
    both the salience information and the original label context needed
    for building hierarchies.
    """

    contour: "BoundaryContour"
    label_context: LabelContextMap


def naive_salience(
    mseg: "MultiSegment",
) -> SaliencePayload:
    """
    Calculates the salience of boundaries based on their frequency of occurrence.

    The salience of each unique boundary time is the number of layers in the
    `MultiSegment` that it appears in. Additionally, this function collects
    all the original labels from each layer for each boundary time to create
    the label context map.

    Args:
        mseg: The `MultiSegment` to analyze.

    Returns:
        A `SaliencePayload` containing:
        - contour: A `BoundaryContour` with salience based on frequency
        - label_context: A mapping from boundary time to all original labels
          from all layers at that time
    """
    # Import here to avoid circular imports
    from .core import BoundaryContour

    # Collect all boundary times and count their frequencies
    time_counts: Counter[float] = Counter()
    # Collect all labels for each time from all layers
    label_context: LabelContextMap = {}

    for layer in mseg.layers:
        for boundary in layer.boundaries:
            time = boundary.time
            time_counts[time] += 1

            # Initialize the label list for this time if not exists
            if time not in label_context:
                label_context[time] = []

            # Add the label from this layer (if it exists)
            if boundary.label is not None:
                label_context[time].append(boundary.label)

    # Create rated boundaries with frequency-based salience
    rated_boundaries = [RatedBoundary(time=time, salience=count) for time, count in time_counts.items()]

    # Create the boundary contour
    contour = BoundaryContour(boundaries=rated_boundaries)

    return SaliencePayload(contour=contour, label_context=label_context)
