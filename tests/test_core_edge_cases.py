"""Short tests for core.py edge cases."""

import pytest

from bnl import core


def test_segment_creation_edge_cases():
    """Test creating a Segment with fewer than two boundaries."""
    with pytest.raises(ValueError, match="requires at least two boundaries"):
        core.Segment(boundaries=[core.Boundary(0)])


def test_boundary_contour_creation_edge_cases():
    """Test creating a BoundaryContour with fewer than two boundaries."""
    with pytest.raises(ValueError, match="requires at least two boundaries"):
        core.BoundaryContour(boundaries=[core.RatedBoundary(0, salience=1)])


def test_multisegment_creation_edge_cases():
    """Test creating a MultiSegment with no layers."""
    with pytest.raises(ValueError, match="cannot be empty"):
        core.MultiSegment(layers=[])

    # Test with layers that contain no boundaries
    with pytest.raises(ValueError, match="at least two boundaries"):
        core.MultiSegment(layers=[core.Segment(boundaries=[])])


def test_hierarchy_creation_edge_cases():
    """Test creating a Hierarchy with fewer than two boundaries."""
    with pytest.raises(ValueError, match="requires at least two boundaries"):
        core.Hierarchy(boundaries=[core.LeveledBoundary(0, ancestry=["a"])])


def test_leveled_boundary_salience_override():
    """Ensure a LeveledBoundary's salience is always its level."""
    # Pass in a different salience; it should be ignored.
    lb = core.LeveledBoundary(time=1.0, ancestry=["a", "b"], salience=99)
    assert lb.level == 2
    assert lb.salience == 2.0


def test_hierarchy_to_multisegment_no_new_layers():
    """
    Tests converting a hierarchy where no sub-layers can be formed.
    e.g., all boundaries are at level 1.
    """
    hier = core.Hierarchy(
        boundaries=[
            core.LeveledBoundary(time=0, ancestry=["L1-a"]),
            core.LeveledBoundary(time=4, ancestry=["L1-a"]),
        ]
    )
    mseg = hier.to_multisegment()
    assert len(mseg.layers) == 1
    assert len(mseg.layers[0].boundaries) == 2


def test_hierarchy_to_multisegment_single_boundary_at_level():
    """
    Tests a hierarchy where a higher level has only one boundary,
    which is not enough to form a valid segment.
    """
    hier = core.Hierarchy(
        boundaries=[
            core.LeveledBoundary(time=0, ancestry=["L1-a"]),
            core.LeveledBoundary(time=4, ancestry=["L1-a"]),
            core.LeveledBoundary(time=2, ancestry=["L1-a", "L2-b"]),  # Just one L2
        ]
    )
    mseg = hier.to_multisegment()
    # Should only produce the L1 segment, as L2 doesn't have >= 2 boundaries.
    assert len(mseg.layers) == 1
    assert len(mseg.layers[0].boundaries) == 3


def test_multisegment_from_json():
    """Tests creating a MultiSegment from a JSON-like structure."""

    json_data = [
        [[[0, 2], [2, 4]], ["A", "B"]],  # Layer 1
        [[[0, 1], [1, 2], [2, 3], [3, 4]], ["a", "b", "c", "d"]],  # Layer 2
    ]
    mseg = core.MultiSegment.from_json(json_data)
    assert len(mseg.layers) == 2
    assert len(mseg.layers[0].boundaries) == 3  # 0, 2, 4
    assert len(mseg.layers[1].boundaries) == 5  # 0, 1, 2, 3, 4

    # Test with empty data
    with pytest.raises(ValueError, match="No valid segments could be created"):
        core.MultiSegment.from_json([])

    # Test with malformed layer
    json_malformed = [[[[0, 2]]], [[[0, 1], [1, 2]], ["a", "b"]]]
    mseg_malformed = core.MultiSegment.from_json(json_malformed)
    assert len(mseg_malformed.layers) == 1  # Should skip the bad layer


def test_multisegment_from_jams():
    """Tests creating a MultiSegment from a JAMS annotation."""
    import jams

    anno = jams.Annotation("multi_segment")
    anno.append(time=0, duration=2.0, value={"level": 0, "label": "A"})
    anno.append(time=2, duration=2.0, value={"level": 0, "label": "B"})
    anno.append(time=0, duration=4.0, value={"level": 1, "label": "X"})

    mseg = core.MultiSegment.from_jams(anno)
    assert len(mseg.layers) == 2
    assert len(mseg.layers[0].boundaries) == 3  # 0, 2, 4
    assert len(mseg.layers[1].boundaries) == 2  # 0, 4

    # Test wrong namespace
    anno.namespace = "segment"
    with pytest.raises(ValueError, match="Expected 'multi_segment'"):
        core.MultiSegment.from_jams(anno)

    # Test no levels by removing the level key from all observations
    anno.namespace = "multi_segment"
    for obs in anno.data:
        del obs.value["level"]
    with pytest.raises(ValueError, match="No valid segments"):
        core.MultiSegment.from_jams(anno)
