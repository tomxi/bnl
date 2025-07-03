"""Operations and transformations for BNL data structures."""

from typing import TYPE_CHECKING, Any, List, Tuple, Dict

# Import necessary classes from .core
from .core import Hierarchy, ProperHierarchy, RatedBoundaries
# Segmentation and TimeSpan might not be directly needed here, but good for context if ops grow.

# if TYPE_CHECKING: # This was the old way, direct import is better for runtime.
#     from .core import Hierarchy


def to_monotonic(hierarchy: Hierarchy) -> ProperHierarchy: # Return ProperHierarchy
    """
    Casts a Hierarchy to be monotonic by creating a ProperHierarchy.

    This baseline strategy derives depths for boundaries based on the
    coarsest layer index in the input hierarchy where each boundary appears.
    """
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("Input must be a bnl.core.Hierarchy object.")

    if not hierarchy.layers: # Should not happen due to Hierarchy validation if __post_init__ ran
        # This case should ideally be prevented by Hierarchy's own validation.
        # If it can occur (e.g. Hierarchy object created skipping validation), handle it.
        return ProperHierarchy.from_rated_boundaries(RatedBoundaries(), name=hierarchy.name)

    boundary_depths: Dict[float, int] = {}

    for layer_idx, layer in enumerate(hierarchy.layers):
        for ts in layer.boundaries: # .boundaries is Tuple[float, ...]
            if ts not in boundary_depths:
                boundary_depths[ts] = layer_idx
            else:
                boundary_depths[ts] = min(boundary_depths[ts], layer_idx)

    if not boundary_depths:
        # All layers in the hierarchy were empty of unique boundaries or hierarchy itself was empty.
        # (e.g., if all layers are Segmentations made from a single boundary point, like [0.0],
        # they would each have boundary (0.0,). If all layers are identical in this way).
        # Or if hierarchy.layers was empty initially (though Hierarchy validation should prevent).
        return ProperHierarchy.from_rated_boundaries(RatedBoundaries(), name=hierarchy.name)

    rated_events_list: List[Tuple[float, int]] = []
    for ts in sorted(boundary_depths.keys()): # Sort timestamps
        rated_events_list.append((ts, boundary_depths[ts]))

    rated_b = RatedBoundaries(events=tuple(rated_events_list))

    proper_h = ProperHierarchy.from_rated_boundaries(rated_b, name=hierarchy.name)

    return proper_h


def boundary_salience(hierarchy: Hierarchy, r: float = 2.0) -> Any: # Input type hint updated
    """Calculates the boundary salience curve for a hierarchy."""
    return None
