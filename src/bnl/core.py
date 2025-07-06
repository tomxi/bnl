"""Core data structures for boundaries-and-labels."""

import numpy as np

__all__ = []


def _validate_time(time: int | float | np.number) -> float:
    """Validates and rounds a time value."""
    if not isinstance(time, int | float | np.number):
        raise TypeError(f"Time must be a number, not {type(time).__name__}.")
    if time < 0:
        raise ValueError("Time cannot be negative.")
    return float(np.round(time, 4))


# region: Point-like Objects


# endregion

# region: Span-like Objects (Containers)


@dataclass(frozen=True)
class TimeSpan:
    """The abstract concept of a time interval."""


# endregion
