"""Hierarchy class for representing hierarchical segmentations."""

from dataclasses import dataclass, field
from typing import List, Optional

from .segment import Segment


@dataclass
class Hierarchy:
    """A hierarchical segmentation composed of multiple levels of segments.
    
    Attributes:
        levels: A list of segment levels, where each level is a list of Segments.
               Levels should be ordered from coarsest to finest.
        name: Optional name for the hierarchy.
    """
    levels: List[List[Segment]] = field(default_factory=list)
    name: Optional[str] = None
    
    @property
    def depth(self) -> int:
        """Return the number of levels in the hierarchy."""
        return len(self.levels)
    
    def get_level(self, index: int) -> List[Segment]:
        """Get a specific level of the hierarchy."""
        if 0 <= index < len(self.levels):
            return self.levels[index]
        raise IndexError(f"Level {index} out of range (0-{len(self.levels)-1})")
    
    def add_level(self, segments: List[Segment]) -> None:
        """Add a new level to the hierarchy."""
        if not segments:
            return
        self.levels.append(segments)
    
    def __repr__(self) -> str:
        level_strs = [f"Level {i}: {len(level)} segments" 
                     for i, level in enumerate(self.levels)]
        return f"Hierarchy({self.name or 'unnamed'}, levels=[{', '.join(level_strs)}])"
