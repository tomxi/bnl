"""Basic tests for the core module."""

from bnl.core import Segment, Hierarchy


def test_segment_creation():
    """Test basic segment creation and properties."""
    seg = Segment(start=1.0, end=2.5, label="A", confidence=0.9)
    
    assert seg.start == 1.0
    assert seg.end == 2.5
    assert seg.label == "A"
    assert seg.confidence == 0.9
    assert seg.duration() == 1.5
    assert str(seg) == "Segment(1.00-2.50, 'A')"


def test_segment_operations():
    """Test segment operations."""
    seg1 = Segment(1.0, 3.0, "A")
    seg2 = Segment(2.0, 4.0, "B")
    
    assert seg1.contains(1.5) is True
    assert seg1.contains(3.0) is False  # end is exclusive
    assert seg1.overlaps(seg2) is True


def test_hierarchy():
    """Test basic hierarchy operations."""
    # Create a simple two-level hierarchy
    level1 = [
        Segment(0.0, 2.5, "A"),
        Segment(2.5, 5.0, "B")
    ]
    
    level2 = [
        Segment(0.0, 1.0, "A1"),
        Segment(1.0, 2.5, "A2"),
        Segment(2.5, 4.0, "B1"),
        Segment(4.0, 5.0, "B2")
    ]
    
    hierarchy = Hierarchy(levels=[level1, level2], name="test")
    
    assert hierarchy.depth == 2
    assert len(hierarchy.get_level(0)) == 2
    assert len(hierarchy.get_level(1)) == 4
    assert "test" in str(hierarchy)
