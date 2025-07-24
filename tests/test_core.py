from dataclasses import FrozenInstanceError

import pytest

from bnl import core


class TestBoundaries:
    """Tests for point-like objects (Boundaries)."""

    def test_boundary_init_and_compare(self):
        b1 = core.Boundary(1.2345677)
        b2 = core.Boundary(1.2345679899)
        b3 = core.Boundary(1.599)
        assert b1 == b2
        assert b1 < b3
        assert repr(b1) == "B(1.23457)"

    def test_rated_boundary_init(self):
        rb = core.RatedBoundary(time=1.5, salience=10.5)
        assert rb.time == 1.5
        assert rb.salience == 10.5
        assert repr(rb) == "RB(1.5, 10.50)"

    def test_rated_boundary_comparison(self):
        rb1 = core.RatedBoundary(1.0, 5.0)
        rb2 = core.RatedBoundary(1.0, 10.0)
        rb3 = core.RatedBoundary(2.0, 5.0)
        assert rb1 < rb2
        assert rb2 < rb3
        assert rb1 < rb3

    def test_leveled_boundary_init(self):
        lb = core.LeveledBoundary(time=2.5, level=5)
        assert lb.time == 2.5
        assert lb.level == 5
        assert lb.salience == 5.0
        assert repr(lb) == "LB(2.5, 5)"

    def test_leveled_boundary_init_invalid_level(self):
        with pytest.raises(ValueError, match="`level` must be a positive integer."):
            core.LeveledBoundary(time=1.0, level=0)
        with pytest.raises(ValueError, match="`level` must be a positive integer."):
            core.LeveledBoundary(time=1.0, level=-1)
        with pytest.raises(ValueError, match="`level` must be a positive integer."):
            core.LeveledBoundary(time=1.0, level=1.5)  # type: ignore

    def test_leveled_boundary_frozen(self):
        lb = core.LeveledBoundary(time=1.0, level=1)
        with pytest.raises(FrozenInstanceError):
            lb.level = 2


class TestTimeSpan:
    def test_timespan_init(self):
        start = core.Boundary(1.0)
        end = core.Boundary(5.0)
        ts = core.TimeSpan(start, end, "test")
        assert ts.start == start
        assert ts.end == end
        assert ts.name == "test"
        assert ts.duration == 4.0
        assert repr(ts) == "TS(B(1.0)-B(5.0), test)"
        assert str(ts) == "test"

    def test_timespan_init_default_name(self):
        ts = core.TimeSpan(core.Boundary(1.0), core.Boundary(2.0))
        assert ts.name == "[1.00-2.00]"
        assert str(ts) == "[1.00-2.00]"

    def test_timespan_init_invalid_duration(self):
        with pytest.raises(ValueError, match="TimeSpan must have a non-zero, positive duration."):
            core.TimeSpan(core.Boundary(2.0), core.Boundary(1.0))
        with pytest.raises(ValueError, match="TimeSpan must have a non-zero, positive duration."):
            core.TimeSpan(core.Boundary(1.0), core.Boundary(1.0))


class TestSegment:
    def test_segment_init(self):
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
        for sec in seg:
            assert sec.name

    def test_segment_init_errors(self):
        with pytest.raises(ValueError, match="A Segment requires at least two boundaries."):
            core.Segment.from_bs([core.Boundary(0)], ["A"])
        boundaries = [core.Boundary(0), core.Boundary(1)]
        with pytest.raises(ValueError, match="Number of labels must be one less than"):
            core.Segment.from_bs(boundaries, ["A", "B"])

    def test_segment_from_methods(self):
        itvls = [[0.0, 1.0], [1.0, 2.5]]
        labels = ["A", "B"]
        seg = core.Segment.from_itvls(itvls, labels)
        assert len(seg.bs) == 3
        assert seg.bs[0].time == 0.0
        assert seg.bs[1].time == 1.0
        assert seg.bs[2].time == 2.5
        assert seg.labels == ["A", "B"]

    def test_multisegment_init(self):
        s1 = core.Segment.from_itvls([[0, 2], [2, 4]], ["A", "B"], name="S1")
        s2 = core.Segment.from_itvls([[0, 1], [1, 4]], ["a", "b"], name="S2")
        mseg = core.MultiSegment(layers=[s1, s2], name="test_mseg")
        assert mseg.name == "test_mseg"
        assert len(mseg) == 2
        assert mseg[0].name == "S1"
        assert mseg.start.time == 0
        assert mseg.end.time == 4

    def test_segment_scrub_labels(self):
        s = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="TestSeg")
        scrubbed_s = s.scrub_labels()
        assert scrubbed_s.labels == ["", ""]
        scrubbed_s_custom = s.scrub_labels(replace_with="x")
        assert scrubbed_s_custom.labels == ["x", "x"]

    def test_segment_align(self):
        s = core.Segment.from_bs([1, 2, 3], ["a", "b"])
        span = core.TimeSpan(core.Boundary(0.5), core.Boundary(3.5))
        aligned_s = s.align(span)
        assert aligned_s.start == span.start
        assert aligned_s.end == span.end
        assert aligned_s.bs == [core.Boundary(0.5), core.Boundary(2), core.Boundary(3.5)]
        s_simple = core.Segment.from_bs([1, 3], ["a"])
        aligned_simple = s_simple.align(span)
        assert aligned_simple.bs == [span.start, span.end]
        invalid_span = core.TimeSpan(core.Boundary(2.5), core.Boundary(3.5))
        with pytest.raises(ValueError):
            s.align(invalid_span)

    def test_segment_plot_api(self):
        s = core.Segment.from_bs([0, 2], ["A"], name="TestSeg")
        fig = s.plot()
        assert fig.data
        assert fig.layout.title.text == "TestSeg"


class TestMultiSegment:
    def test_multisegment_init_errors(self):
        with pytest.raises(ValueError, match="MultiSegment must contain at least one"):
            core.MultiSegment(layers=[])
        s1 = core.Segment.from_itvls([[0, 2], [2, 4]], ["A", "B"])
        s2_bad_start = core.Segment.from_itvls([[0.1, 1], [1, 4]], ["a", "b"])
        s2_bad_end = core.Segment.from_itvls([[0, 1], [1, 4.1]], ["a", "b"])
        fixed_start = core.MultiSegment(layers=[s1, s2_bad_start])
        fixed_end = core.MultiSegment(layers=[s1, s2_bad_end])
        assert fixed_start.start.time == 0
        assert fixed_end.end.time == 4.1
        for layer in fixed_start:
            assert layer.start.time == 0
        for layer in fixed_end:
            assert layer.end.time == 4.1

    def test_multisegment_from_json(self):
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

    def test_multisegment_prune_layers(self):
        s1 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L01")
        s2 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L02")
        s3 = core.Segment.from_bs([0, 1.5, 2], ["c", "d"], name="L03")
        ms = core.MultiSegment(layers=[s1, s2, s3], name="TestMS")
        pruned_ms = ms.prune_layers(relabel=True)
        assert len(pruned_ms.layers) == 2
        assert pruned_ms.layers[0].name == "L01"
        assert pruned_ms.layers[1].name == "L02"
        assert pruned_ms.layers[0].bs == s1.bs
        assert pruned_ms.layers[1].bs == s3.bs
        pruned_ms_no_relabel = ms.prune_layers(relabel=False)
        assert len(pruned_ms_no_relabel.layers) == 2
        assert pruned_ms_no_relabel.layers[0].name == "L01"
        assert pruned_ms_no_relabel.layers[1].name == "L03"

    def test_ms_plot_api(self):
        s1 = core.Segment.from_bs([0, 2], ["A"], name="L01")
        s2 = core.Segment.from_bs([0, 1, 2], ["a", "b"], name="L02")
        s3 = core.Segment.from_bs([0, 1.5, 2], ["c", "d"], name="L03")
        ms = core.MultiSegment(layers=[s1, s2, s3], name="TestMS")
        fig = ms.plot()
        assert fig.layout.title.text == "TestMS"


class TestContours:
    """Tests for contour-like objects (BoundaryContour, BoundaryHierarchy)."""

    def test_boundary_contour_init(self):
        # This is unsorted, let's see if it gets sorted.
        boundaries = [core.RatedBoundary(1, 1), core.RatedBoundary(0, 5), core.RatedBoundary(2, 2)]
        contour = core.BoundaryContour(bs=boundaries, name="test_contour")
        assert contour.name == "test_contour"
        assert len(contour) == 1
        assert contour[0].time == 1 and contour[0].salience == 1
        assert contour.start.time == 0
        assert contour.end.time == 2
        for b in contour:
            assert b <= contour.end and b >= contour.start

    def test_boundary_contour_init_error(self):
        with pytest.raises(ValueError, match="A BoundaryContour requires at least two boundaries."):
            core.BoundaryContour(bs=[core.RatedBoundary(0, 1)], name="test contour")

    def test_bc_chain_apis(self):
        boundaries = [core.RatedBoundary(1, 1), core.RatedBoundary(0, 5), core.RatedBoundary(2, 2)]
        contour = core.BoundaryContour(bs=boundaries, name="test_contour")
        fig = contour.clean().level().plot()
        assert fig.layout.title.text == "test_contour"

    def test_boundary_hierarchy_init(self):
        boundaries = [
            core.LeveledBoundary(1, 1),
            core.LeveledBoundary(0, 2),
            core.LeveledBoundary(2, 1),
        ]
        hier = core.BoundaryHierarchy(bs=boundaries, name="test_hier")
        assert hier.name == "test_hier"
        assert len(hier) == 1
        assert hier[0].time == 1 and hier[0].level == 1
        assert hier.start.time == 0
        assert hier.end.time == 2

    def test_boundary_hierarchy_init_error(self):
        with pytest.raises(ValueError, match="A BoundaryContour requires at least two boundaries."):
            core.BoundaryHierarchy(bs=[core.LeveledBoundary(0, 1)], name="test hierarchy")

    def test_boundary_hierarchy_type_validation(self):
        with pytest.raises(TypeError, match="All boundaries must be LeveledBoundary instances"):
            core.BoundaryHierarchy(
                bs=[
                    core.LeveledBoundary(0, 1),
                    core.RatedBoundary(1, 2.0),
                ],
                name="test hierarchy",
            )
        with pytest.raises(TypeError, match="All boundaries must be LeveledBoundary instances"):
            core.BoundaryHierarchy(
                bs=[
                    core.LeveledBoundary(0, 1),
                    core.Boundary(1),
                ],
                name="test hierarchy",
            )

    def test_boundaryhierarchy_post_init_type_error(self):
        with pytest.raises(TypeError):
            core.BoundaryHierarchy(bs=[core.RatedBoundary(0, 1), core.RatedBoundary(1, 1)])

    def test_boundaryhierarchy_to_ms(self):
        bh = core.BoundaryHierarchy(
            bs=[core.LeveledBoundary(0, 2), core.LeveledBoundary(1, 1), core.LeveledBoundary(2, 2)],
            name="TestBH",
        )
        ms = bh.to_ms()
        assert isinstance(ms, core.MultiSegment)
        assert len(ms.layers) == 2
        assert ms.layers[0].name == "L01"
        assert ms.layers[0].bs == [core.Boundary(0), core.Boundary(2)]
        assert ms.layers[1].name == "L02"
        assert ms.layers[1].bs == [core.Boundary(0), core.Boundary(1), core.Boundary(2)]
        assert ms.name == "TestBH Monotonic MS"
        bh_no_name = core.BoundaryHierarchy(
            bs=[core.LeveledBoundary(0, 1), core.LeveledBoundary(1, 1)]
        )
        ms_no_name = bh_no_name.to_ms()
        assert ms_no_name.name == "BoundaryHierarchy Monotonic MS"
