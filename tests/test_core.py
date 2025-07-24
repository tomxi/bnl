from dataclasses import FrozenInstanceError

import pytest

from bnl import core

# region: Point-like Objects


def test_boundary_init_and_compare():
    b1 = core.Boundary(1.2345677)
    b2 = core.Boundary(1.2345679899)
    b3 = core.Boundary(1.599)
    assert b1 == b2
    assert b1 < b3
    assert repr(b1) == "B(1.23457)"


def test_rated_boundary_init():
    rb = core.RatedBoundary(time=1.5, salience=10.5)
    assert rb.time == 1.5
    assert rb.salience == 10.5
    assert repr(rb) == "RB(1.5, 10.50)"


def test_rated_boundary_comparison():
    rb1 = core.RatedBoundary(1.0, 5.0)
    rb2 = core.RatedBoundary(1.0, 10.0)
    rb3 = core.RatedBoundary(2.0, 5.0)
    assert rb1 < rb2
    assert rb2 < rb3
    assert rb1 < rb3


def test_leveled_boundary_init():
    lb = core.LeveledBoundary(time=2.5, level=5)
    assert lb.time == 2.5
    assert lb.level == 5
    assert lb.salience == 5.0
    assert repr(lb) == "LB(2.5, 5)"


def test_leveled_boundary_init_invalid_level():
    with pytest.raises(ValueError, match="`level` must be a positive integer."):
        core.LeveledBoundary(time=1.0, level=0)
    with pytest.raises(ValueError, match="`level` must be a positive integer."):
        core.LeveledBoundary(time=1.0, level=-1)
    with pytest.raises(ValueError, match="`level` must be a positive integer."):
        core.LeveledBoundary(time=1.0, level=1.5)  # type: ignore


def test_leveled_boundary_frozen():
    lb = core.LeveledBoundary(time=1.0, level=1)
    with pytest.raises(FrozenInstanceError):
        lb.level = 2


# endregion


# region: Span-like Objects (Containers)


def test_timespan_init():
    start = core.Boundary(1.0)
    end = core.Boundary(5.0)
    ts = core.TimeSpan(start, end, "test")
    assert ts.start == start
    assert ts.end == end
    assert ts.name == "test"
    assert ts.duration == 4.0
    assert repr(ts) == "TS(B(1.0)-B(5.0), test)"
    assert str(ts) == "test"


def test_timespan_init_default_name():
    ts = core.TimeSpan(core.Boundary(1.0), core.Boundary(2.0))
    assert ts.name == "[1.00-2.00]"
    assert str(ts) == "[1.00-2.00]"


def test_timespan_init_invalid_duration():
    with pytest.raises(ValueError, match="TimeSpan must have a non-zero, positive duration."):
        core.TimeSpan(core.Boundary(2.0), core.Boundary(1.0))
    with pytest.raises(ValueError, match="TimeSpan must have a non-zero, positive duration."):
        core.TimeSpan(core.Boundary(1.0), core.Boundary(1.0))


def test_segment_init():
    boundaries = [core.Boundary(0), core.Boundary(1), core.Boundary(2)]
    labels = ["A", "B"]
    seg = core.Segment.from_bs(boundaries, labels, "test_seg")
    assert seg.name == "test_seg"
    assert seg.start.time == 0
    assert seg.end.time == 2
    assert len(seg) == 2
    assert seg[0].name == "A"
    assert seg[1].duration == 1.0
    assert seg.sections[1].end.time == 2.0


def test_segment_init_errors():
    with pytest.raises(ValueError, match="A Segment requires at least two boundaries."):
        core.Segment.from_bs([core.Boundary(0)], ["A"])
    boundaries = [core.Boundary(0), core.Boundary(1)]
    with pytest.raises(ValueError, match="Number of labels must be one less than"):
        core.Segment.from_bs(boundaries, ["A", "B"])
    unsorted_boundaries = [core.Boundary(1), core.Boundary(0)]
    with pytest.raises(ValueError, match="Boundaries must be sorted by time."):
        core.Segment.from_bs(unsorted_boundaries, ["A"])


def test_segment_from_methods():
    itvls = [[0.0, 1.0], [1.0, 2.5]]
    labels = ["A", "B"]
    seg = core.Segment.from_itvls(itvls, labels)
    assert len(seg.bs) == 3
    assert seg.bs[0].time == 0.0
    assert seg.bs[1].time == 1.0
    assert seg.bs[2].time == 2.5
    assert seg.labels == ["A", "B"]


def test_multisegment_init():
    s1 = core.Segment.from_itvls([[0, 2], [2, 4]], ["A", "B"], name="S1")
    s2 = core.Segment.from_itvls([[0, 1], [1, 4]], ["a", "b"], name="S2")
    mseg = core.MultiSegment(layers=[s1, s2], name="test_mseg")
    assert mseg.name == "test_mseg"
    assert len(mseg) == 2
    assert mseg[0].name == "S1"
    assert mseg.start.time == 0
    assert mseg.end.time == 4


def test_multisegment_init_errors():
    with pytest.raises(ValueError, match="MultiSegment must contain at least one"):
        core.MultiSegment(layers=[])
    s1 = core.Segment.from_itvls([[0, 2], [2, 4]], ["A", "B"])
    s2_bad_start = core.Segment.from_itvls([[0.1, 1], [1, 4]], ["a", "b"])
    s2_bad_end = core.Segment.from_itvls([[0, 1], [1, 4.1]], ["a", "b"])
    fixed_start = core.MultiSegment(layers=[s1, s2_bad_start])
    fixed_end = core.MultiSegment(layers=[s1, s2_bad_end])
    # The start and end times are fixed to the common start and end of the layers.
    assert fixed_start.start.time == 0.1
    assert fixed_end.end.time == 4


def test_multisegment_from_json():
    json_data = [
        [[[0, 2], [2, 4]], ["A", "B"]],
        [[[0, 1], [1, 2], [2, 3], [3, 4]], ["a", "b", "c", "d"]],
    ]
    mseg = core.MultiSegment.from_json(json_data)
    assert len(mseg.layers) == 2
    assert mseg.name == "JSON Annotation"
    assert mseg.layers[0].name == "L01"
    assert len(mseg.layers[0].bs) == 3
    assert len(mseg.layers[1].bs) == 5



# endregion


def test_boundary_contour_init():
    boundaries = [core.RatedBoundary(1, 1), core.RatedBoundary(0, 5), core.RatedBoundary(2, 2)]
    contour = core.BoundaryContour(bs=boundaries, name="test_contour")
    assert contour.name == "test_contour"
    assert len(contour) == 1
    # Check the effective (internal) boundary
    assert contour[0].time == 1 and contour[0].salience == 1
    # Check start and end boundaries
    assert contour.start.time == 0
    assert contour.end.time == 2


def test_boundary_contour_init_error():
    with pytest.raises(ValueError, match="A BoundaryContour requires at least two boundaries."):
        core.BoundaryContour(bs=[core.RatedBoundary(0, 1)], name='test contour')


def test_boundary_hierarchy_init():
    boundaries = [core.LeveledBoundary(1, 1), core.LeveledBoundary(0, 2), core.LeveledBoundary(2, 1)]
    hier = core.BoundaryHierarchy(bs=boundaries, name="test_hier")
    assert hier.name == "test_hier"
    assert len(hier) == 1
    # Check the effective (internal) boundary
    assert hier[0].time == 1 and hier[0].level == 1
    # Check start and end boundaries
    assert hier.start.time == 0
    assert hier.end.time == 2


def test_boundary_hierarchy_init_error():
    with pytest.raises(ValueError, match="A BoundaryContour requires at least two boundaries."):
        core.BoundaryHierarchy(bs=[core.LeveledBoundary(0, 1)], name="test hierarchy")


def test_multisegment_prune_layers():
    s1 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L01")
    s2 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L02")
    s3 = core.Segment.from_bs([0, 1.5, 2], ["c", "d"], name="L03")
    ms = core.MultiSegment(layers=[s1, s2, s3], name="TestMS")

    # Test pruning with relabeling
    pruned_ms = ms.prune_layers(relabel=True)
    assert len(pruned_ms.layers) == 2
    assert pruned_ms.layers[0].name == "L01"
    assert pruned_ms.layers[1].name == "L02"
    assert pruned_ms.layers[0].bs == s1.bs
    assert pruned_ms.layers[1].bs == s3.bs

    # Test pruning without relabeling
    pruned_ms_no_relabel = ms.prune_layers(relabel=False)
    assert len(pruned_ms_no_relabel.layers) == 2
    assert pruned_ms_no_relabel.layers[0].name == "L01"
    assert pruned_ms_no_relabel.layers[1].name == "L03"


def test_boundary_hierarchy_type_validation():
    """Test that BoundaryHierarchy only accepts LeveledBoundary instances."""
    # Should raise TypeError when passed RatedBoundary instead of LeveledBoundary
    with pytest.raises(TypeError, match="All boundaries must be LeveledBoundary instances"):
        core.BoundaryHierarchy(
            bs=[
                core.LeveledBoundary(0, 1),
                core.RatedBoundary(1, 2.0),  # This should cause TypeError
            ],
            name="test hierarchy",
        )

    # Should raise TypeError when passed regular Boundary
    with pytest.raises(TypeError, match="All boundaries must be LeveledBoundary instances"):
        core.BoundaryHierarchy(
            bs=[
                core.LeveledBoundary(0, 1),
                core.Boundary(1),  # This should cause TypeError
            ],
            name="test hierarchy"
        )


def test_segment_scrub_labels():
    """Tests the scrub_labels method of Segment."""
    s = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="TestSeg")
    scrubbed_s = s.scrub_labels()
    assert scrubbed_s.labels == ["", ""]
    scrubbed_s_custom = s.scrub_labels(replace_with="x")
    assert scrubbed_s_custom.labels == ["x", "x"]


def test_segment_align():
    """Tests the align method of Segment."""
    s = core.Segment.from_bs([1, 2, 3], ["a", "b"])
    span = core.TimeSpan(core.Boundary(0.5), core.Boundary(3.5))
    aligned_s = s.align(span)
    assert aligned_s.start == span.start
    assert aligned_s.end == span.end
    assert aligned_s.bs == [core.Boundary(0.5), core.Boundary(2), core.Boundary(3.5)]

    # Test align with only 2 boundaries
    s_simple = core.Segment.from_bs([1, 3], ["a"])
    aligned_simple = s_simple.align(span)
    assert aligned_simple.bs == [span.start, span.end]

    # Test align with invalid span
    invalid_span = core.TimeSpan(core.Boundary(2.5), core.Boundary(3.5))
    with pytest.raises(ValueError):
        s.align(invalid_span)


def test_multisegment_find_span():
    """Tests the find_span method of MultiSegment."""
    s1 = core.Segment.from_bs([0, 1], ["a"])
    s2 = core.Segment.from_bs([0.5, 1.5], ["b"])
    # This will align the layers to the common span [0.5, 1.0]
    ms = core.MultiSegment(layers=[s1, s2])

    union_span = ms.find_span(mode="union")
    assert union_span.start.time == 0.5
    assert union_span.end.time == 1.0

    common_span = ms.find_span(mode="common")
    assert common_span.start.time == 0.5
    assert common_span.end.time == 1.0

    with pytest.raises(ValueError):
        ms.find_span(mode="invalid_mode")


def test_boundaryhierarchy_post_init_type_error():
    """Tests that BoundaryHierarchy raises TypeError for incorrect boundary types."""
    with pytest.raises(TypeError):
        core.BoundaryHierarchy(
            bs=[core.RatedBoundary(0, 1), core.RatedBoundary(1, 1)]
        )


def test_boundaryhierarchy_to_ms():
    """Tests the to_ms conversion for BoundaryHierarchy."""
    bh = core.BoundaryHierarchy(
        bs=[core.LeveledBoundary(0, 2), core.LeveledBoundary(1, 1), core.LeveledBoundary(2, 2)],
        name="TestBH"
    )
    ms = bh.to_ms()
    assert isinstance(ms, core.MultiSegment)
    assert len(ms.layers) == 2
    assert ms.layers[0].name == "L01"  # Coarsest layer (level 2)
    assert ms.layers[0].bs == [core.Boundary(0), core.Boundary(2)]
    assert ms.layers[1].name == "L02"  # Finest layer (level 1)
    assert ms.layers[1].bs == [core.Boundary(0), core.Boundary(1), core.Boundary(2)]
    assert ms.name == "TestBH"

    # Test default name
    bh_no_name = core.BoundaryHierarchy(
        bs=[core.LeveledBoundary(0, 1), core.LeveledBoundary(1, 1)]
    )
    ms_no_name = bh_no_name.to_ms()
    assert ms_no_name.name == "BoundaryContour"

def test_multisegment_prune_layers():
    s1 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L01")
    s2 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L02") # Identical to s1
    s3 = core.Segment.from_bs([0, 1.5, 2], ["c", "d"], name="L03") # Different
    ms = core.MultiSegment(layers=[s1, s2, s3], name="TestMS")

    # Test pruning with relabeling
    pruned_ms = ms.prune_layers(relabel=True)
    assert len(pruned_ms.layers) == 2
    assert pruned_ms.layers[0].name == "L01" # Check relabeling
    assert pruned_ms.layers[1].name == "L02"
    assert pruned_ms.layers[0].bs == s1.bs
    assert pruned_ms.layers[1].bs == s3.bs

    # Test pruning without relabeling
    pruned_ms_no_relabel = ms.prune_layers(relabel=False)
    assert len(pruned_ms_no_relabel.layers) == 2
    assert pruned_ms_no_relabel.layers[0].name == "L01" # Original names preserved
    assert pruned_ms_no_relabel.layers[1].name == "L03"