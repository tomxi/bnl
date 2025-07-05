import numpy as np
import pytest

from bnl.core import (
    Boundary,
    Hierarchy,
    ProperHierarchy,
    RatedBoundaries,
    RatedBoundary,
    Segmentation,
    TimeSpan,
)
from bnl.strategies import (
    BoundaryGroupingStrategy,
    DirectSynthesisStrategy,
    LevelGroupingStrategy,
)


# Test Boundary
def test_boundary_init():
    b = Boundary(1.23456)
    assert b.time == 1.2346  # check rounding
    assert b.label is None


def test_boundary_with_label():
    b = Boundary(2.0, label="event")
    assert b.label == "event"


def test_boundary_type_validation():
    with pytest.raises(TypeError, match="Time must be a number"):
        Boundary("not a time")  # type: ignore


def test_boundary_value_validation():
    with pytest.raises(ValueError, match="Time cannot be negative"):
        Boundary(-1.0)


def test_boundary_sorting():
    b1 = Boundary(1.0)
    b2 = Boundary(2.0)
    b3 = Boundary(1.0, label="another")
    assert b1 < b2
    assert b1 == b3  # Comparison is only on time


# Test TimeSpan
def test_timespan_init():
    b1 = Boundary(1.0)
    ts = TimeSpan(b1, 2.0, "A")
    assert ts.start == b1
    assert ts.duration == 2.0
    assert ts.label == "A"
    assert ts.end == Boundary(3.0)
    assert ts.end.time == 3.0


def test_timespan_no_label():
    b1 = Boundary(1.0)
    ts = TimeSpan(b1, 0.5)
    assert ts.label is None


def test_timespan_post_init_validation():
    b1 = Boundary(1.0)
    with pytest.raises(ValueError, match="Duration must be positive"):
        TimeSpan(b1, 0)
    with pytest.raises(ValueError, match="Duration must be positive"):
        TimeSpan(b1, -1.0)


def test_timespan_validation():
    b1 = Boundary(2.0)
    with pytest.raises(ValueError):
        TimeSpan(b1, -1.0)


# Test Segmentation
def test_segmentation_init():
    b1, b2, b3 = Boundary(0), Boundary(1.5), Boundary(3.0)
    seg = Segmentation(boundaries=[b1, b2, b3], labels=["verse", "chorus"], name="song_structure")
    assert seg.start == b1
    assert seg.end == b3
    assert seg.duration == 3.0
    assert len(seg) == 2
    assert seg.segments[0] == TimeSpan(b1, 1.5, "verse")
    assert seg.segments[1] == TimeSpan(b2, 1.5, "chorus")


def test_segmentation_unsorted_boundaries():
    b1, b2, b3 = Boundary(0), Boundary(3.0), Boundary(1.5)
    seg = Segmentation(boundaries=[b1, b2, b3], labels=["A", "B"], name="song_structure")
    assert seg.boundaries[1].time == 1.5
    assert seg.boundaries[2].time == 3.0


def test_segmentation_len_and_getitem():
    b1 = Boundary(0.0)
    b2 = Boundary(3.0)
    b3 = Boundary(5.0)
    seg = Segmentation(boundaries=[b1, b2, b3], labels=["verse", "chorus"], name="Test")
    assert len(seg.segments) == 2
    assert seg.segments[0] == TimeSpan(b1, 3.0, "verse")
    assert seg.segments[1] == TimeSpan(b2, 2.0, "chorus")


def test_segmentation_from_boundaries():
    times = [0, 1.5, 3]
    labels = ["A", "B"]
    seg = Segmentation.from_boundaries(times, labels)
    assert len(seg) == 2
    assert seg[0].duration == 1.5


def test_segmentation_from_intervals():
    intervals = np.array([[0, 1.5], [1.5, 3]])
    labels = ["A", "B"]
    seg = Segmentation.from_intervals(intervals, labels)
    assert len(seg) == 2
    assert seg[0].duration == 1.5


def test_segmentation_errors():
    with pytest.raises(ValueError, match="at least one boundary"):
        Segmentation(boundaries=[], labels=[])
    with pytest.raises(ValueError, match="Number of labels"):
        Segmentation(boundaries=[Boundary(0), Boundary(1)], labels=[])


# Test Hierarchy
def test_hierarchy_init():
    s1 = Segmentation.from_boundaries([0, 2, 4], ["A", "B"])
    s2 = Segmentation.from_boundaries([0, 1, 2, 3, 4], ["a", "b", "c", "d"])
    h = Hierarchy([s1, s2], name="TestHier")
    assert len(h) == 2
    assert h.duration == 4.0


def test_hierarchy_alignment_error():
    s1 = Segmentation.from_boundaries([0, 2, 4], ["A", "B"])
    s2 = Segmentation.from_boundaries([0, 1, 5], ["a", "b"])  # End time mismatch
    with pytest.raises(ValueError, match="All layers must span the same time range"):
        Hierarchy([s1, s2])


# Test ProperHierarchy
def test_proper_hierarchy_monotonic_validation():
    # This should work
    s1 = Segmentation.from_boundaries([0, 4])
    s2 = Segmentation.from_boundaries([0, 2, 4])
    s3 = Segmentation.from_boundaries([0, 1, 2, 3, 4])
    ph = ProperHierarchy([s1, s2, s3])
    assert ph is not None

    # This should fail
    s_fail1 = Segmentation.from_boundaries([0, 2, 4])
    s_fail2 = Segmentation.from_boundaries([0, 3, 4])  # Not a superset
    with pytest.raises(ValueError, match="not a subset"):
        ProperHierarchy([s_fail1, s_fail2])


def test_rated_boundaries_fluent_api():
    """Tests that the fluent API on RatedBoundaries calls strategies correctly."""

    class MockGrouping(BoundaryGroupingStrategy):
        def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
            # Test that this gets called by adding a boundary with the same salience
            return boundaries + [RatedBoundary(99, 1)]

    class MockLeveling(LevelGroupingStrategy):
        def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
            # Test that this gets called by checking the event count
            return DirectSynthesisStrategy().quantize(boundaries)

    # 1. Initial data
    initial_events = [RatedBoundary(1, 1)]
    rb = RatedBoundaries(events=initial_events, start_time=0.0, end_time=100.0)

    # 2. Chain the calls
    final_ph = rb.group_boundaries(MockGrouping()).quantize_level(MockLeveling())

    # 3. Assertions
    assert isinstance(final_ph, ProperHierarchy)
    # Layer 0 should have the global start/end
    assert len(final_ph[0].boundaries) == 2
    # Layer 1 should have the global start/end + the two event times
    assert len(final_ph[1].boundaries) == 4  # 0, 1, 99, 100


# Test string representations
def test_timespan_str_and_repr():
    """Test the string representations of TimeSpan."""
    b1 = Boundary(0.0)
    ts = TimeSpan(b1, 1.0, "A")
    assert str(ts) == "TimeSpan([0.00s-1.00s], 1.00s: A)"
    assert repr(ts) == f"TimeSpan(start={b1!r}, duration=1.0, label='A')"


def test_rated_boundary_repr():
    rb = RatedBoundary(1.0, salience=5, label="C5")
    assert repr(rb) == "RatedBoundary(time=1.0, salience=5, label='C5')"


def test_segmentation_repr():
    b1 = Boundary(0.0)
    b2 = Boundary(1.0)
    seg = Segmentation([b1, b2], ["A"], name="MySeg")
    assert repr(seg) == "Segmentation(name='MySeg', 1 segments, duration=1.00s)"


def test_hierarchy_repr():
    s1 = Segmentation.from_boundaries([0, 2, 4], ["A", "B"])
    h = Hierarchy([s1], name="MyHier")
    assert repr(h) == "Hierarchy(name='MyHier', 1 layers, duration=4.00s)"
