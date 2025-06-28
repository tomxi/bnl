"""Short tests for core.py edge cases."""

import pytest

from bnl import core


def test_hierarchy_empty():
    """Test empty hierarchy."""
    h = core.Hierarchy([])
    assert len(h) == 0


def test_hierarchy_properties():
    """Test basic properties."""
    layers = [core.Segmentation.from_intervals([[0, 5], [5, 10]])]
    h = core.Hierarchy(layers=layers)
    assert h.start == 0
    assert h.end == 10
    assert len(h) == 1


def test_hierarchy_from_jams_wrong_namespace():
    """Test from_jams with wrong namespace."""
    from unittest.mock import Mock

    mock_ann = Mock()
    mock_ann.namespace = "wrong"

    with pytest.raises(ValueError, match="Expected 'multi_segment'"):
        core.Hierarchy.from_jams(mock_ann)
