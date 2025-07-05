import pytest

from bnl.core import (
    Boundary,
    MultiSegment,  # Was Hierarchy
    Hierarchy,     # Was ProperHierarchy
    RatedBoundaries,
    RatedBoundary,
    LeveledBoundary, # New
    BoundaryContour, # New
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

    with pytest.raises(ValueError, match="empty times"):
        Segmentation.from_boundaries([])
    with pytest.raises(ValueError, match="empty intervals"):
        Segmentation.from_intervals([])


def test_multisegment_creation_and_validation():  # Renamed test
    """Tests MultiSegment creation and time alignment validation."""
    s1 = Segmentation.from_boundaries([0, 2, 4], ["A", "B"])
    s2 = Segmentation.from_boundaries([0, 1, 2, 3, 4], ["a", "b", "c", "d"])
    ms = MultiSegment(start=s1.start, duration=s1.duration, layers=[s1, s2], label="MyMultiSeg") # Renamed h to ms
    assert ms.label == "MyMultiSeg"
    assert len(ms.layers) == 2
    assert ms.start.time == 0
    assert ms.end.time == 4
    assert ms.duration == 4
    assert len(ms) == 2
    assert ms[0] == s1

    s3_misaligned = Segmentation.from_boundaries([0, 1, 5], ["a", "b"])
    with pytest.raises(ValueError, match="All layers must span the same time range"):
        MultiSegment(start=s1.start, duration=s1.duration, layers=[s1, s3_misaligned])


def test_hierarchy_monotonicity_and_leveledboundaries(): # Renamed test
    """Tests the monotonicity enforcement and LeveledBoundary usage of Hierarchy."""
    # This should work
    # For Hierarchy, boundaries must be LeveledBoundary.
    # DirectSynthesisStrategy is expected to produce LeveledBoundaries.

    # Create RatedBoundaries that DirectSynthesisStrategy can convert to LeveledBoundary
    rb_events_s1 = [RatedBoundary(0, rate=1), RatedBoundary(4, rate=1)]
    rated_boundaries_s1 = RatedBoundaries(events=rb_events_s1, start_time=0, end_time=4)
    s1_from_synth = DirectSynthesisStrategy().quantize(rated_boundaries_s1).layers[0]

    rb_events_s2 = [RatedBoundary(0, rate=2), RatedBoundary(2, rate=2), RatedBoundary(4, rate=2)]
    # To ensure s2's rates are different for a multi-level hierarchy if needed by test logic
    # but for this simple case, rate=2 is fine.
    # For proper nesting, the *times* must nest. Levels are inherent to LeveledBoundary.
    # Let's use distinct rates to form distinct levels for a clearer test of Hierarchy construction.
    rated_boundaries_s2 = RatedBoundaries(events=[RatedBoundary(0, rate=2), RatedBoundary(2,rate=1), RatedBoundary(4, rate=2)], start_time=0, end_time=4)
    # The DirectSynthesisStrategy will create layers based on unique rates.
    # For simplicity, we'll construct segmentations with LeveledBoundary directly for this test.

    lb0 = LeveledBoundary(0, level=1)
    lb1 = LeveledBoundary(1, level=2)
    lb2 = LeveledBoundary(2, level=1)
    lb3 = LeveledBoundary(3, level=2)
    lb4 = LeveledBoundary(4, level=1)

    s1 = Segmentation(start=lb0, duration=4, boundaries=[lb0, lb4])
    s2 = Segmentation(start=lb0, duration=4, boundaries=[lb0, lb2, lb4])
    s3 = Segmentation(start=lb0, duration=4, boundaries=[lb0, lb1, lb2, lb3, lb4])

    h_good = Hierarchy(start=lb0, duration=4, layers=[s1, s2, s3]) # Renamed ph_good to h_good
    assert len(h_good.layers) == 3
    assert isinstance(h_good.layers[0].boundaries[0], LeveledBoundary)

    # Test LeveledBoundary validation in Hierarchy's __post_init__
    s_fail_type_boundaries = [Boundary(0), Boundary(4)]
    s_fail_type = Segmentation(start=Boundary(0), duration=4, boundaries=s_fail_type_boundaries)
    with pytest.raises(TypeError, match="All boundaries in a Hierarchy must be LeveledBoundary"):
        Hierarchy(start=Boundary(0), duration=4, layers=[s_fail_type])

    # This should fail (s2 is not a superset of s1's boundaries)
    # Using LeveledBoundary for these tests too
    s_fail1_lb0, s_fail1_lb2, s_fail1_lb4 = LeveledBoundary(0,1), LeveledBoundary(2,1), LeveledBoundary(4,1)
    s_fail2_lb0, s_fail2_lb3, s_fail2_lb4 = LeveledBoundary(0,1), LeveledBoundary(3,1), LeveledBoundary(4,1)

    s_fail1 = Segmentation(start=s_fail1_lb0, duration=4, boundaries=[s_fail1_lb0, s_fail1_lb2, s_fail1_lb4])
    s_fail2 = Segmentation(start=s_fail2_lb0, duration=4, boundaries=[s_fail2_lb0, s_fail2_lb3, s_fail2_lb4])

    with pytest.raises(ValueError, match="Monotonicity violation"):
        Hierarchy(start=s_fail1_lb0, duration=4, layers=[s_fail1, s_fail2])


def test_rated_boundaries_fluent_api():
    """Tests the fluent API for grouping and quantizing RatedBoundaries."""

    class MockGrouping(BoundaryGroupingStrategy):
        def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
            # Ensure rate is positive for LeveledBoundary if used by DirectSynthesisStrategy
            return boundaries + [RatedBoundary(99, rate=1)]

    class MockLeveling(LevelGroupingStrategy):
        # This should return the new Hierarchy (formerly ProperHierarchy)
        def quantize(self, boundaries: RatedBoundaries) -> Hierarchy:
            return DirectSynthesisStrategy().quantize(boundaries)

    initial_events = [RatedBoundary(1, rate=1)] # Use rate
    rb = RatedBoundaries(events=initial_events, start_time=0.0, end_time=100.0)
    final_h = rb.group_boundaries(MockGrouping()).quantize_level(MockLeveling()) # Renamed final_ph

    assert isinstance(final_h, Hierarchy) # Check for new Hierarchy
    # DirectSynthesisStrategy creates LeveledBoundary, rates become levels.
    # Level 1 contains start, end, and two boundaries (initial + mocked)
    # The structure of layers depends on unique rates. If all rates are 1, one layer.
    assert len(final_h.layers) > 0 # Should have at least one layer
    if final_h.layers:
        assert len(final_h.layers[0].boundaries) >= 2 # start and end, plus events
        assert isinstance(final_h.layers[0].boundaries[0], LeveledBoundary)


def test_string_representations():
    """Tests the __str__ and __repr__ methods of core objects."""
    b = Boundary(0.0)
    ts = TimeSpan(b, 1.0, "A")
    assert str(ts) == "TimeSpan([0.00s-1.00s], 1.00s: A)"
    assert repr(ts) == f"TimeSpan(start={b!r}, duration=1.0, label='A')"

    rb = RatedBoundary(1.0, rate=5, label="C5") # Use rate
    assert repr(rb) == "RatedBoundary(time=1.00, rate=5.00, label='C5')" # Use rate

    lb = LeveledBoundary(1.0, level=5, label="L5")
    assert repr(lb) == "LeveledBoundary(time=1.00, level=5, label='L5')"


    seg = Segmentation(start=b, duration=1.0, boundaries=[b, Boundary(1.0)], label="MySeg")
    assert repr(seg) == "Segmentation(label='MySeg', 1 segments, duration=1.00s)"

    # Create a segmentation with LeveledBoundary for hierarchy tests
    lb0 = LeveledBoundary(0, level=1)
    lb4 = LeveledBoundary(4, level=1)
    seg_for_h = Segmentation(start=lb0, duration=4.0, boundaries=[lb0, lb4])


    ms = MultiSegment(start=b, duration=4.0, layers=[seg_for_h], label="MyMultiSeg") # Renamed h to ms
    assert repr(ms) == "MultiSegment(label='MyMultiSeg', 1 layers, duration=4.00s)" # Check MultiSegment

    h = Hierarchy(start=lb0, duration=4.0, layers=[seg_for_h], label="MyHier") # Renamed ph to h, use LeveledBoundary
    h_repr = repr(h)
    # Example of a more robust check for the repr of Hierarchy
    assert "Hierarchy" in h_repr
    assert f"start={lb0!r}" in h_repr
    assert "duration=4.0" in h_repr
    assert "label='MyHier'" in h_repr
    assert "layers=" in h_repr


def test_plotting_functions():
    """Tests the plot methods of core objects."""
    import matplotlib.pyplot as plt

    # 1. Test TimeSpan.plot
    b = Boundary(0.0)
    ts = TimeSpan(b, 1.0, "A")

    # Test without providing ax
    ax_ts_new = ts.plot()
    assert ax_ts_new is not None
    plt.close(ax_ts_new.figure)

    # Test with providing ax
    fig, ax = plt.subplots()
    ax_ts_existing = ts.plot(ax=ax)
    assert ax_ts_existing is ax
    plt.close(fig)

    # 2. Test Segmentation.plot
    seg = Segmentation.from_boundaries([0, 1.5, 3], ["A", "B"])

    # Test without providing ax
    ax_seg_new = seg.plot()
    assert ax_seg_new is not None
    plt.close(ax_seg_new.figure)

    # Test with providing ax
    fig, ax = plt.subplots()
    ax_seg_existing = seg.plot(ax=ax)
    assert ax_seg_existing is ax
    plt.close(fig)

    # Test with color_map
    fig, ax = plt.subplots()
    ax_seg_color = seg.plot(ax=ax, color_map={"A": "red", "B": "blue"})
    assert ax_seg_color is ax
    plt.close(fig)

    # 3. Test MultiSegment.plot (formerly Hierarchy.plot)
    s1_ms = Segmentation.from_boundaries([0, 2, 4], ["A", "B"]) # Renamed s1 to s1_ms
    s2_ms = Segmentation.from_boundaries([0, 1, 2, 3, 4], ["a", "b", "c", "d"]) # Renamed s2 to s2_ms
    ms = MultiSegment(start=s1_ms.start, duration=s1_ms.duration, layers=[s1_ms, s2_ms], label="MyMultiSeg") # Renamed h to ms

    # Test without providing ax
    ax_ms_new = ms.plot() # Renamed ax_h_new
    assert ax_ms_new is not None
    plt.close(ax_ms_new.figure)

    # Test with providing ax
    fig, ax = plt.subplots()
    ax_ms_existing = ms.plot(ax=ax) # Renamed ax_h_existing
    assert ax_ms_existing is ax
    plt.close(fig)

    # 4. Test Hierarchy.plot (formerly ProperHierarchy.plot)
    # Hierarchy requires LeveledBoundary instances
    lb0_h = LeveledBoundary(0, level=1)
    lb2_h = LeveledBoundary(2, level=1)
    lb4_h = LeveledBoundary(4, level=1)

    s1_h = Segmentation(start=lb0_h, duration=4, boundaries=[lb0_h, lb4_h]) # Renamed s1_ph to s1_h
    s2_h = Segmentation(start=lb0_h, duration=4, boundaries=[lb0_h, lb2_h, lb4_h]) # Renamed s2_ph to s2_h

    h = Hierarchy(start=lb0_h, duration=s1_h.duration, layers=[s1_h, s2_h]) # Renamed ph to h

    # Test without providing ax
    ax_h_new = h.plot() # Renamed ax_ph_new
    assert ax_h_new is not None
    plt.close(ax_h_new.figure)

    # Test with providing ax
    fig, ax = plt.subplots()
    ax_h_existing = h.plot(ax=ax) # Renamed ax_ph_existing
    assert ax_h_existing is ax
    plt.close(fig)
