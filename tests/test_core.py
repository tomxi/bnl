import pytest

from bnl.core import (
    Boundary,
    BoundaryContour,
    Hierarchy,
    LeveledBoundary,
    MultiSegment,
    RatedBoundary,
    Segment,
    TimeSpan,
)


def test_boundary_creation_and_ordering():
    """Tests basic Boundary creation and time-based ordering."""
    b_ok = Boundary(1.23456, "event")
    assert b_ok.time == 1.2346
    assert b_ok.label == "event"

    with pytest.raises(TypeError):
        Boundary("not a time")  # type: ignore
    with pytest.raises(ValueError):
        Boundary(-1.0)

    b1 = Boundary(1.0)
    b2 = Boundary(2.0)
    assert b1 < b2


def test_rated_boundary_ordering():
    """Tests that RatedBoundary orders by time, then salience."""
    b1 = RatedBoundary(1.0, salience=5)
    b2 = RatedBoundary(2.0, salience=5)
    b3 = RatedBoundary(1.0, salience=10)
    assert b1 < b2
    assert b1 < b3
    assert sorted([b3, b1]) == [b1, b3]


def test_leveled_boundary_properties():
    """Tests derived properties of LeveledBoundary."""
    lb = LeveledBoundary(1.0, ancestry=["a", "b", "c"])
    assert lb.level == 3
    assert lb.salience == 3.0


def test_timespan_creation_and_validation():
    """Tests basic TimeSpan creation and validation."""
    b1 = Boundary(1.0)
    ts = TimeSpan(b1, 2.0)
    assert ts.start == b1
    assert ts.duration == 2.0
    assert ts.end.time == 3.0

    with pytest.raises(ValueError):
        TimeSpan(b1, -1.0)


def test_segment_creation_and_properties():
    """Tests Segment creation, sorting, and derived properties."""
    b1, b2, b3 = Boundary(0, "A"), Boundary(3.0), Boundary(1.5, "B")
    seg = Segment(boundaries=[b1, b3, b2])  # Pass unsorted
    assert seg.boundaries[0].time == 0
    assert seg.boundaries[1].time == 1.5
    assert seg.boundaries[2].time == 3.0
    assert seg.start.time == 0
    assert seg.duration == 3.0

    with pytest.raises(ValueError, match="at least two boundaries"):
        Segment(boundaries=[b1])


def test_boundary_contour_creation():
    """Tests BoundaryContour creation and sorting."""
    rb1 = RatedBoundary(0.0, salience=1)
    rb2 = RatedBoundary(2.0, salience=3)
    rb3 = RatedBoundary(1.0, salience=2)
    contour = BoundaryContour(boundaries=[rb1, rb3, rb2])
    assert contour.boundaries[0].time == 0
    assert contour.boundaries[1].time == 1
    assert contour.boundaries[2].time == 2
    assert contour.duration == 2.0

    with pytest.raises(ValueError, match="at least two boundaries"):
        BoundaryContour(boundaries=[rb1])


def test_multisegment_creation():
    """Tests MultiSegment creation and derived properties."""
    seg1 = Segment(boundaries=[Boundary(0), Boundary(2)])
    seg2 = Segment(boundaries=[Boundary(1), Boundary(4)])
    mseg = MultiSegment(layers=[seg1, seg2])
    assert len(mseg.layers) == 2
    assert mseg.start.time == 0
    assert mseg.end.time == 4
    assert mseg.duration == 4.0

    with pytest.raises(ValueError, match="cannot be empty"):
        MultiSegment(layers=[])


def test_hierarchy_creation():
    """Tests Hierarchy creation and sorting."""
    lb1 = LeveledBoundary(time=0, ancestry=["a"])
    lb3 = LeveledBoundary(time=2, ancestry=["a", "b", "c"])
    lb2 = LeveledBoundary(time=1, ancestry=["a", "b"])
    hier = Hierarchy(boundaries=[lb1, lb3, lb2])
    assert hier.boundaries[0].time == 0
    assert hier.boundaries[1].time == 1
    assert hier.boundaries[2].time == 2
    assert hier.duration == 2.0

    with pytest.raises(ValueError, match="at least two boundaries"):
        Hierarchy(boundaries=[lb1])


def test_hierarchy_to_multisegment():
    """Tests the conversion from a Hierarchy to a MultiSegment."""
    lb1 = LeveledBoundary(time=0, ancestry=["L1-a"], salience=1)
    lb2 = LeveledBoundary(time=4, ancestry=["L1-a"], salience=1)
    lb3 = LeveledBoundary(time=2, ancestry=["L1-a", "L2-b"], salience=2)
    lb4 = LeveledBoundary(time=1, ancestry=["L1-a", "L2-b", "L3-c"], salience=3)
    lb5 = LeveledBoundary(time=3, ancestry=["L1-a", "L2-b", "L3-c"], salience=3)
    hier = Hierarchy(boundaries=[lb1, lb2, lb3, lb4, lb5])

    mseg = hier.to_multisegment()
    assert len(mseg.layers) == 3

    # Level 1 Segment: Boundaries from levels 1, 2, 3
    level1_seg = mseg.layers[0]
    assert len(level1_seg.boundaries) == 5
    assert {b.time for b in level1_seg.boundaries} == {0, 1, 2, 3, 4}
    assert {b.label for b in level1_seg.boundaries} == {"L1-a"}

    # Level 2 Segment: Boundaries from levels 2, 3
    level2_seg = mseg.layers[1]
    assert len(level2_seg.boundaries) == 3
    assert {b.time for b in level2_seg.boundaries} == {1, 2, 3}
    assert {b.label for b in level2_seg.boundaries} == {"L2-b"}

    # Level 3 Segment: Boundaries from level 3
    level3_seg = mseg.layers[2]
    assert len(level3_seg.boundaries) == 2
    assert {b.time for b in level3_seg.boundaries} == {1, 3}
    assert {b.label for b in level3_seg.boundaries} == {"L3-c"}


def test_empty_hierarchy_conversion():
    """Test converting a hierarchy with insufficient boundaries for layers."""
    # Hierarchy with boundaries at the same level, but not enough for a L2 segment
    hier = Hierarchy(
        boundaries=[
            LeveledBoundary(time=0, ancestry=["L1-a"]),
            LeveledBoundary(time=4, ancestry=["L1-a"]),
        ]
    )
    mseg = hier.to_multisegment()
    assert len(mseg.layers) == 1
    assert len(mseg.layers[0].boundaries) == 2


def test_multisegment_to_boundary_contour():
    """Tests converting a MultiSegment to a BoundaryContour."""
    # Layer 1 has boundaries at 0, 2, 4
    # Layer 2 has boundaries at 0, 4, 8
    # Frequencies: 0 (2), 2 (1), 4 (2), 8 (1)
    s1 = Segment.from_intervals([[0, 2], [2, 4]])
    s2 = Segment.from_intervals([[0, 4], [4, 8]])
    mseg = MultiSegment(layers=[s1, s2])

    contour = mseg.to_contour(method="frequency")
    assert isinstance(contour, BoundaryContour)
    assert len(contour.boundaries) == 4

    # Check that saliences are correct
    salience_map = {b.time: b.salience for b in contour.boundaries}
    assert salience_map[0] == 2
    assert salience_map[2] == 1
    assert salience_map[4] == 2
    assert salience_map[8] == 1

    with pytest.raises(ValueError, match="Unsupported"):
        mseg.to_contour(method="invalid_method")
