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


def test_boundary_creation_and_validation():
    """Tests basic Boundary creation, validation, and sorting."""
    b_ok = Boundary(1.23456, "event")
    assert b_ok.time == 1.2346, "Time should be rounded"
    assert b_ok.label == "event"

    with pytest.raises(TypeError):
        Boundary("not a time")  # type: ignore
    with pytest.raises(ValueError):
        Boundary(-1.0)

    b1 = Boundary(1.0)
    b2 = Boundary(2.0)
    b3 = Boundary(1.0, label="another")
    assert b1 < b2
    assert b1 == b3, "Boundary comparison should only use time"


def test_timespan_creation_and_validation():
    """Tests basic TimeSpan creation and validation."""
    b1 = Boundary(1.0)
    ts = TimeSpan(b1, 2.0, "A")
    assert ts.start == b1
    assert ts.duration == 2.0
    assert ts.label == "A"
    assert ts.end.time == 3.0, "End time should be calculated correctly"

    with pytest.raises(ValueError, match="Duration must be positive"):
        TimeSpan(b1, 0)


def test_segmentation_creation_and_properties():
    """Tests Segmentation creation, including property access and internal sorting."""
    b1, b2, b3 = Boundary(0, "A"), Boundary(3.0), Boundary(1.5, "B")
    seg = Segmentation(start=b1, duration=3.0, boundaries=[b1, b2, b3], label="test_seg")
    assert seg.label == "test_seg"
    assert seg.boundaries[0].time == 0 and seg.boundaries[2].time == 3.0, "Boundaries should be sorted"
    assert len(seg) == 2, "Length should be the number of segments"
    assert seg[0].duration == 1.5


def test_segmentation_factories():
    """Tests the `from_boundaries` and `from_intervals` factory methods."""
    seg_b = Segmentation.from_boundaries([0, 1.5, 3], ["A", "B"])
    assert len(seg_b) == 2
    assert seg_b[0].duration == 1.5

    seg_i = Segmentation.from_intervals([[0, 1.5], [1.5, 3]], ["A", "B"])
    assert len(seg_i) == 2
    assert seg_i[0].duration == 1.5


def test_segmentation_edge_cases():
    """Tests error handling for invalid Segmentation definitions."""
    with pytest.raises(ValueError, match="at least one boundary"):
        Segmentation(start=Boundary(0), duration=1, boundaries=[])


def test_hierarchy_creation_and_validation():
    """Tests Hierarchy creation and time alignment validation."""
    s1 = Segmentation.from_boundaries([0, 2, 4], ["A", "B"])
    s2 = Segmentation.from_boundaries([0, 1, 2, 3, 4], ["a", "b", "c", "d"])
    h = Hierarchy(start=s1.start, duration=s1.duration, layers=[s1, s2], label="MyHier")
    assert h.label == "MyHier"
    assert len(h.layers) == 2

    s3_misaligned = Segmentation.from_boundaries([0, 1, 5], ["a", "b"])
    with pytest.raises(ValueError, match="All layers must span the same time range"):
        Hierarchy(start=s1.start, duration=s1.duration, layers=[s1, s3_misaligned])


def test_proper_hierarchy_monotonicity():
    """Tests the monotonicity enforcement of ProperHierarchy."""
    # This should work
    s1 = Segmentation.from_boundaries([0, 4])
    s2 = Segmentation.from_boundaries([0, 2, 4])
    s3 = Segmentation.from_boundaries([0, 1, 2, 3, 4])
    ph_good = ProperHierarchy(start=s1.start, duration=s1.duration, layers=[s1, s2, s3])
    assert len(ph_good.layers) == 3

    # This should fail (s2 is not a superset of s1's boundaries)
    s_fail1 = Segmentation.from_boundaries([0, 2, 4])
    s_fail2 = Segmentation.from_boundaries([0, 3, 4])
    with pytest.raises(ValueError, match="Monotonicity violation"):
        ProperHierarchy(start=s_fail1.start, duration=s_fail1.duration, layers=[s_fail1, s_fail2])


def test_rated_boundaries_fluent_api():
    """Tests the fluent API for grouping and quantizing RatedBoundaries."""

    class MockGrouping(BoundaryGroupingStrategy):
        def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
            return boundaries + [RatedBoundary(99, 1)]

    class MockLeveling(LevelGroupingStrategy):
        def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
            return DirectSynthesisStrategy().quantize(boundaries)

    initial_events = [RatedBoundary(1, 1)]
    rb = RatedBoundaries(events=initial_events, start_time=0.0, end_time=100.0)
    final_ph = rb.group_boundaries(MockGrouping()).quantize_level(MockLeveling())

    assert isinstance(final_ph, ProperHierarchy)
    # Level 0 (salience >= 1) contains start, end, and two boundaries (initial + mocked)
    assert len(final_ph[0].boundaries) == 4
    assert len(final_ph.layers) == 1


def test_string_representations():
    """Tests the __str__ and __repr__ methods of core objects."""
    b = Boundary(0.0)
    ts = TimeSpan(b, 1.0, "A")
    assert str(ts) == "TimeSpan([0.00s-1.00s], 1.00s: A)"
    assert repr(ts) == f"TimeSpan(start={b!r}, duration=1.0, label='A')"

    rb = RatedBoundary(1.0, salience=5, label="C5")
    assert repr(rb) == "RatedBoundary(time=1.00, salience=5.00, label='C5')"

    seg = Segmentation(start=b, duration=1.0, boundaries=[b, Boundary(1.0)], label="MySeg")
    assert repr(seg) == "Segmentation(label='MySeg', 1 segments, duration=1.00s)"

    h = Hierarchy(start=b, duration=4.0, layers=[seg], label="MyHier")
    assert repr(h) == "Hierarchy(label='MyHier', 1 layers, duration=4.00s)"

    ph = ProperHierarchy(start=b, duration=4.0, layers=[seg], label="MyProperHier")
    ph_repr = repr(ph)
    assert "ProperHierarchy" in ph_repr
    assert "label='MyProperHier'" in ph_repr
