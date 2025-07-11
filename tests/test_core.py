from dataclasses import FrozenInstanceError

import pytest

from bnl import core

# region: Point-like Objects


def test_boundary_init():
    b = core.Boundary(1.234567)
    assert b.time == 1.23457  # Rounded to 5 decimal places
    assert repr(b) == "B(1.23457)"


def test_boundary_comparison():
    b1 = core.Boundary(1.0)
    b2 = core.Boundary(2.0)
    assert b1 < b2
    assert b1 != b2


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
    assert ts.name == ""
    assert str(ts) == "[1.00-2.00]"


def test_timespan_init_invalid_duration():
    with pytest.raises(ValueError, match="TimeSpan must have a non-zero, positive duration."):
        core.TimeSpan(core.Boundary(2.0), core.Boundary(1.0))
    with pytest.raises(ValueError, match="TimeSpan must have a non-zero, positive duration."):
        core.TimeSpan(core.Boundary(1.0), core.Boundary(1.0))


def test_segment_init():
    boundaries = [core.Boundary(0), core.Boundary(1), core.Boundary(2)]
    labels = ["A", "B"]
    seg = core.Segment(boundaries, labels, "test_seg")
    assert seg.name == "test_seg"
    assert seg.start.time == 0
    assert seg.end.time == 2
    assert len(seg) == 2
    assert seg[0].name == "A"
    assert seg[1].duration == 1.0
    assert seg.sections[1].end.time == 2.0


def test_segment_init_errors():
    with pytest.raises(ValueError, match="A Segment requires at least two boundaries."):
        core.Segment([core.Boundary(0)], ["A"])
    boundaries = [core.Boundary(0), core.Boundary(1)]
    with pytest.raises(ValueError, match="Number of labels must be one less than"):
        core.Segment(boundaries, ["A", "B"])
    unsorted_boundaries = [core.Boundary(1), core.Boundary(0)]
    with pytest.raises(ValueError, match="Boundaries must be sorted by time."):
        core.Segment(unsorted_boundaries, ["A"])


def test_segment_from_methods():
    itvls = [[0.0, 1.0], [1.0, 2.5]]
    labels = ["A", "B"]
    seg = core.Segment.from_itvls(itvls, labels)
    assert len(seg.boundaries) == 3
    assert seg.boundaries[0].time == 0.0
    assert seg.boundaries[1].time == 1.0
    assert seg.boundaries[2].time == 2.5
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
    with pytest.raises(ValueError, match="All layers must have the same start time"):
        core.MultiSegment([s1, s2_bad_start])
    with pytest.raises(ValueError, match="All layers must have the same end time"):
        core.MultiSegment([s1, s2_bad_end])


def test_multisegment_from_json():
    json_data = [
        [[[0, 2], [2, 4]], ["A", "B"]],
        [[[0, 1], [1, 2], [2, 3], [3, 4]], ["a", "b", "c", "d"]],
    ]
    mseg = core.MultiSegment.from_json(json_data)
    assert len(mseg.layers) == 2
    assert mseg.name == "JSON Annotation"
    assert mseg.layers[0].name == "L00"
    assert len(mseg.layers[0].boundaries) == 3
    assert len(mseg.layers[1].boundaries) == 5


def test_multisegment_align_layers():
    s1 = core.Segment.from_itvls([[0, 5]], ["A"])
    s2 = core.Segment.from_itvls([[1, 3], [3, 6]], ["b", "c"])
    s3 = core.Segment.from_itvls([[2, 4]], ["d"])

    aligned_layers = core.MultiSegment.align_layers([s1, s2, s3])
    assert len(aligned_layers) == 3

    min_start = 0
    max_end = 6

    for layer in aligned_layers:
        assert layer.start.time == min_start
        assert layer.end.time == max_end

    # Test original layers are not modified
    assert s1.start.time == 0 and s1.end.time == 5

    # Test empty list
    assert core.MultiSegment.align_layers([]) == []


def test_plotting_methods_return_axes():
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    fig, ax = plt.subplots()

    ts = core.TimeSpan(core.Boundary(0), core.Boundary(1))
    ax_ts = ts.plot(ax)
    assert isinstance(ax_ts, Axes)

    seg = core.Segment.from_itvls([[0, 1]], ["A"])
    ax_seg = seg.plot(ax)
    assert isinstance(ax_seg, Axes)

    mseg = core.MultiSegment([seg])
    ax_mseg = mseg.plot(ax)
    assert isinstance(ax_mseg, Axes)

    plt.close(fig)


# endregion


def test_boundary_contour_init():
    boundaries = [core.RatedBoundary(1, 1), core.RatedBoundary(0, 5), core.RatedBoundary(2, 2)]
    contour = core.BoundaryContour("test_contour", boundaries)
    assert contour.name == "test_contour"
    assert len(contour) == 3
    assert contour[0].time == 0 and contour[0].salience == 5
    assert contour[1].time == 1 and contour[1].salience == 1
    assert contour[2].time == 2 and contour[2].salience == 2
    assert contour.start.time == 0
    assert contour.end.time == 2


def test_boundary_contour_init_error():
    with pytest.raises(ValueError, match="At least 2 boundaries for a TimeSpan!"):
        core.BoundaryContour("test", [core.RatedBoundary(0, 1)])


def test_boundary_hierarchy_init():
    boundaries = [core.LeveledBoundary(1, 1), core.LeveledBoundary(0, 2), core.LeveledBoundary(2, 1)]
    hier = core.BoundaryHierarchy("test_hier", boundaries)
    assert hier.name == "test_hier"
    assert len(hier) == 3
    assert hier[0].time == 0 and hier[0].level == 2
    assert hier[1].time == 1 and hier[1].level == 1
    assert hier[2].time == 2 and hier[2].level == 1
    assert hier.start.time == 0
    assert hier.end.time == 2


def test_boundary_hierarchy_init_error():
    with pytest.raises(ValueError, match="At least 2 boundaries for a TimeSpan!"):
        core.BoundaryHierarchy("test", [core.LeveledBoundary(0, 1)])
