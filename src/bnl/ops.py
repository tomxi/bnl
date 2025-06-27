"""Operations and transformations for BNL data structures."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Hierarchy


def to_monotonic(hierarchy: "Hierarchy") -> "Hierarchy":
    """Casts a Hierarchy to be monotonic."""
    return hierarchy


def boundary_salience(hierarchy: "Hierarchy", r: float = 2.0) -> Any:
    """Calculates the boundary salience curve for a hierarchy."""
    return None
