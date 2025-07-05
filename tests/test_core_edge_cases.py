"""Short tests for core.py edge cases."""

import jams
import pytest

from bnl import core


def test_segmentation_time_mismatch():
    """Tests that a Segmentation's start/end times must match its boundaries."""
    b1, b2 = core.Boundary(1.0), core.Boundary(2.0)
    # Start time mismatch
    with pytest.raises(ValueError, match="start time"):
        core.Segmentation(start=core.Boundary(0.0), duration=1.0, boundaries=[b1, b2])
    # End time mismatch
    with pytest.raises(ValueError, match="end time"):
        core.Segmentation(start=b1, duration=2.0, boundaries=[b1, b2])


def test_segmentation_from_jams_no_data():
    """Tests `from_jams` behavior with no observation data."""
    anno = jams.Annotation("segment")
    with pytest.raises(ValueError, match="no data and no defined range"):
        core.Segmentation.from_jams(anno)

    # Should succeed if a range is provided
    seg = core.Segmentation.from_jams(anno, start_time=0.0, duration=10.0)
    assert seg.start.time == 0.0
    assert seg.duration == 10.0
    assert len(seg.boundaries) == 2


def test_hierarchy_empty():
    """Test that creating a Hierarchy with no layers raises ValueError."""
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        core.Hierarchy(start=core.Boundary(0), duration=0, layers=[])


def test_hierarchy_from_jams_wrong_namespace():
    """Test from_jams with wrong namespace."""
    from unittest.mock import Mock

    mock_ann = Mock()
    mock_ann.namespace = "wrong"

    with pytest.raises(ValueError, match="Expected 'multi_segment'"):
        core.Hierarchy.from_jams(mock_ann)


def test_hierarchy_from_jams_no_levels():
    """Test from_jams with a multi_segment annotation that has no levels."""
    anno = jams.Annotation("multi_segment")
    # Add an observation without a "level" in its value
    anno.append(time=0, duration=1, value={"label": "A"})
    with pytest.raises(ValueError, match="no levels"):
        core.Hierarchy.from_jams(anno)


def test_hierarchy_from_jams_no_duration_fallback():
    """Tests Hierarchy time range fallback when annotation has no duration."""
    anno = jams.Annotation("multi_segment")
    anno.append(time=0, duration=1.0, value={"level": 0, "label": "a"})
    anno.append(time=1, duration=1.0, value={"level": 0, "label": "b"})
    anno.append(time=0, duration=2.0, value={"level": 1, "label": "c"})
    anno.duration = None  # Explicitly remove duration
    h = core.Hierarchy.from_jams(anno)
    assert h.start.time == 0.0
    assert h.duration == 2.0


def test_hierarchy_from_json_empty():
    """Tests from_json with empty data."""
    with pytest.raises(ValueError, match="empty JSON data"):
        core.Hierarchy.from_json([])


def test_rated_boundary_invalid_salience():
    """Tests that non-numeric salience raises a TypeError."""
    with pytest.raises(TypeError):
        core.RatedBoundary(time=0, salience="invalid")  # type: ignore


def test_proper_hierarchy_from_rated_boundaries():
    """Tests the ProperHierarchy.from_rated_boundaries factory."""
    events = [core.RatedBoundary(1.0, 2), core.RatedBoundary(2.0, 1)]
    ph = core.ProperHierarchy.from_rated_boundaries(events, start_time=0.0, end_time=3.0)
    assert isinstance(ph, core.ProperHierarchy)
    assert len(ph.layers) == 2
    assert ph.start.time == 0.0
    assert ph.duration == 3.0
    # Coarsest layer (salience >= 2) has one event
    assert len(ph[0].boundaries) == 3  # start, event, end
    # Finest layer (salience >= 1) has both events
    assert len(ph[1].boundaries) == 4
