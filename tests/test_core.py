from dataclasses import FrozenInstanceError

import pytest

from bnl import core

# region: Point-like Objects


def test_boundary_init():
    b1 = core.Boundary(1.2345677)
    b2 = core.Boundary(1.2345679899)
    assert b1 == b2
    assert repr(b1) == "B(1.23457)"


def test_boundary_comparison():
    b1 = core.Boundary(1.2345677)
    b2 = core.Boundary(1.599)
    assert b1 < b2


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
    mseg = core.MultiSegment([s1, s2], name="test_mseg")
    assert mseg.name == "test_mseg"
    assert len(mseg) == 2
    assert mseg[0].name == "S1"
    assert mseg.start.time == 0
    assert mseg.end.time == 4


def test_multisegment_init_errors():
    with pytest.raises(ValueError, match="MultiSegment must contain at least one"):
        core.MultiSegment([])
    s1 = core.Segment.from_itvls([[0, 2], [2, 4]], ["A", "B"])
    s2_bad_start = core.Segment.from_itvls([[0.1, 1], [1, 4]], ["a", "b"])
    s2_bad_end = core.Segment.from_itvls([[0, 1], [1, 4.1]], ["a", "b"])
    fixed_start = core.MultiSegment([s1, s2_bad_start])
    fixed_end = core.MultiSegment([s1, s2_bad_end])
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
    contour = core.BoundaryContour("test_contour", boundaries)
    assert contour.name == "test_contour"
    assert len(contour) == 1
    # Check the effective (internal) boundary
    assert contour[0].time == 1 and contour[0].salience == 1
    # Check start and end boundaries
    assert contour.start.time == 0
    assert contour.end.time == 2


def test_boundary_contour_init_error():
    with pytest.raises(ValueError, match="At least 2 boundaries for a TimeSpan!"):
        core.BoundaryContour("test", [core.RatedBoundary(0, 1)])


def test_boundary_hierarchy_init():
    boundaries = [core.LeveledBoundary(1, 1), core.LeveledBoundary(0, 2), core.LeveledBoundary(2, 1)]
    hier = core.BoundaryHierarchy("test_hier", boundaries)
    assert hier.name == "test_hier"
    assert len(hier) == 1
    # Check the effective (internal) boundary
    assert hier[0].time == 1 and hier[0].level == 1
    # Check start and end boundaries
    assert hier.start.time == 0
    assert hier.end.time == 2


def test_boundary_hierarchy_init_error():
    with pytest.raises(ValueError, match="At least 2 boundaries for a TimeSpan!"):
        core.BoundaryHierarchy("test", [core.LeveledBoundary(0, 1)])


def test_boundary_hierarchy_type_validation():
    """Test that BoundaryHierarchy only accepts LeveledBoundary instances."""
    # Should raise TypeError when passed RatedBoundary instead of LeveledBoundary
    with pytest.raises(TypeError, match="All boundaries must be LeveledBoundary instances"):
        core.BoundaryHierarchy(
            "test",
            [
                core.LeveledBoundary(0, 1),
                core.RatedBoundary(1, 2.0),  # This should cause TypeError
            ],
        )

    # Should raise TypeError when passed regular Boundary
    with pytest.raises(TypeError, match="All boundaries must be LeveledBoundary instances"):
        core.BoundaryHierarchy(
            "test",
            [
                core.LeveledBoundary(0, 1),
                core.Boundary(1),  # This should cause TypeError
            ],
        )
