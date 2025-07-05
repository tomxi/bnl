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
    # This tests the MultiSegment's validation, as Hierarchy inherits from it.
    with pytest.raises(ValueError, match="MultiSegment must contain at least one layer."):
        core.Hierarchy(start=core.LeveledBoundary(0, level=1), duration=0, layers=[])


def test_hierarchy_from_jams_wrong_namespace():
    """Test from_jams with wrong namespace."""
    from unittest.mock import Mock

    mock_ann = Mock()
    mock_ann.namespace = "wrong"

    with pytest.raises(ValueError, match="Expected 'multi_segment'"):
        core.MultiSegment.from_jams(mock_ann) # Changed to MultiSegment


def test_multisegment_from_jams_no_levels(): # Renamed test
    """Test from_jams with a multi_segment annotation that has no levels."""
    anno = jams.Annotation("multi_segment")
    # Add an observation without a "level" in its value
    anno.append(time=0, duration=1, value={"label": "A"})
    with pytest.raises(ValueError, match="no levels"):
        core.MultiSegment.from_jams(anno) # Changed to MultiSegment


def test_multisegment_from_jams_no_duration_fallback(): # Renamed test
    """Tests MultiSegment time range fallback when annotation has no duration.""" # Changed
    anno = jams.Annotation("multi_segment")
    anno.append(time=0, duration=1.0, value={"level": 0, "label": "a"})
    anno.append(time=1, duration=1.0, value={"level": 0, "label": "b"})
    anno.append(time=0, duration=2.0, value={"level": 1, "label": "c"}) # This data implies non-monotonicity if directly made into new Hierarchy
    anno.duration = None  # Explicitly remove duration
    ms = core.MultiSegment.from_jams(anno) # Changed to MultiSegment
    assert ms.start.time == 0.0
    assert ms.duration == 2.0 # Max end time is 2.0
    assert len(ms.layers) == 2
    # MultiSegment does not guarantee LeveledBoundary
    assert isinstance(ms.layers[0].boundaries[0], core.Boundary)
    assert not isinstance(ms.layers[0].boundaries[0], core.LeveledBoundary)


def test_multisegment_from_json_empty(): # Renamed test
    """Tests from_json with empty data."""
    with pytest.raises(ValueError, match="empty JSON data"):
        core.MultiSegment.from_json([]) # Changed to MultiSegment


def test_rated_boundary_invalid_rate(): # Renamed test
    """Tests that non-numeric rate raises a TypeError.""" # salience -> rate
    with pytest.raises(TypeError):
        core.RatedBoundary(time=0, rate="invalid")  # type: ignore # salience -> rate


def test_hierarchy_from_rated_boundaries(): # Renamed test
    """Tests the Hierarchy.from_rated_boundaries factory.""" # ProperHierarchy -> Hierarchy
    events = [core.RatedBoundary(1.0, rate=2), core.RatedBoundary(2.0, rate=1)] # salience -> rate
    h = core.Hierarchy.from_rated_boundaries(events, start_time=0.0, end_time=3.0) # ph -> h, ProperHierarchy -> Hierarchy
    assert isinstance(h, core.Hierarchy) # ph -> h, ProperHierarchy -> Hierarchy
    assert len(h.layers) == 2 # Two distinct rate values should create two layers
    assert h.start.time == 0.0
    assert h.duration == 3.0

    # DirectSynthesisStrategy (used by from_rated_boundaries) sorts unique rates descending.
    # So, layer 0 corresponds to rate 2, layer 1 to rate 1.
    # All boundaries within these layers must be LeveledBoundary.

    # Layer 0 (rate 2 -> level 2)
    # Contains events with rate >= 2. Event at 1.0 (rate 2). Plus start/end.
    layer0_boundaries = h.layers[0].boundaries
    assert len(layer0_boundaries) == 3  # start, event at 1.0, end
    assert layer0_boundaries[0].time == 0.0 and layer0_boundaries[0].level == 2
    assert layer0_boundaries[1].time == 1.0 and layer0_boundaries[1].level == 2
    assert layer0_boundaries[2].time == 3.0 and layer0_boundaries[2].level == 2
    assert all(isinstance(b, core.LeveledBoundary) for b in layer0_boundaries)

    # Layer 1 (rate 1 -> level 1)
    # Contains events with rate >= 1. Event at 1.0 (rate 2) and 2.0 (rate 1). Plus start/end.
    layer1_boundaries = h.layers[1].boundaries
    assert len(layer1_boundaries) == 4  # start, event at 1.0, event at 2.0, end
    assert layer1_boundaries[0].time == 0.0 and layer1_boundaries[0].level == 1
    assert layer1_boundaries[1].time == 1.0 and layer1_boundaries[1].level == 2 # Original level was 2
    assert layer1_boundaries[2].time == 2.0 and layer1_boundaries[2].level == 1 # Original level was 1
    assert layer1_boundaries[3].time == 3.0 and layer1_boundaries[3].level == 1
    assert all(isinstance(b, core.LeveledBoundary) for b in layer1_boundaries)
