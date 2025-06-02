"""Segment class for representing labeled time intervals."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """A labeled time interval in a hierarchical segmentation.
    
    Attributes:
        start: Start time of the segment (in seconds).
        end: End time of the segment (in seconds).
        label: Label for the segment.
        confidence: Optional confidence score for the segment.
    """
    start: float
    end: float
    label: str
    confidence: Optional[float] = None
    
    def duration(self) -> float:
        """Return the duration of the segment in seconds."""
        return self.end - self.start
    
    def contains(self, time: float) -> bool:
        """Check if the segment contains the given time point."""
        return self.start <= time < self.end
    
    def overlaps(self, other: 'Segment') -> bool:
        """Check if this segment overlaps with another segment."""
        return (self.start < other.end) and (self.end > other.start)
    
    def __repr__(self) -> str:
        return f"Segment({self.start:.2f}-{self.end:.2f}, '{self.label}')"
