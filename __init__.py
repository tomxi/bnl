"""BNL: A Python library for hierarchical segmentation analysis."""

__version__ = "0.1.0"

# Core functionality
from .core.segment import Segment
from .core.hierarchy import Hierarchy

__all__ = [
    'Segment',
    'Hierarchy',
]
