"""Tests for operations in bnl.ops."""

import pytest

from bnl import (
    core,  # For creating dummy Hierarchy objects
    ops,
)

# It's good practice to also test with type hints if they are complex
# but for these stubs, basic calls are sufficient for coverage.


def test_to_monotonic_stub():
    """Test the to_monotonic stub function."""
    # Create a dummy Hierarchy object.
    # Since Hierarchy's __post_init__ checks for consistent layer start/end times,
    # and Segmentation's __post_init__ checks for contiguous segments,
    # we need to provide valid, albeit simple, structures.
    seg1 = core.Segmentation.from_boundaries([0.0, 2.0, 5.0], ["A1", "A2"])
    seg2 = core.Segmentation.from_boundaries([0.0, 1.0, 3.0, 5.0], ["B1", "B2", "B3"])

    # Ensure layers have same start/end to satisfy Hierarchy post_init
    # The above seg1 and seg2 already have start=0, end=5

    dummy_hierarchy = core.Hierarchy(layers=[seg1, seg2])

    try:
        result = ops.to_monotonic(dummy_hierarchy)
        # The stub returns the input hierarchy directly
        assert result is dummy_hierarchy
    except Exception as e:
        pytest.fail(f"ops.to_monotonic raised an exception: {e}")


def test_boundary_salience_stub():
    """Test the boundary_salience stub function."""
    seg1 = core.Segmentation.from_boundaries([0.0, 2.0, 5.0], ["A1", "A2"])
    seg2 = core.Segmentation.from_boundaries([0.0, 1.0, 3.0, 5.0], ["B1", "B2", "B3"])
    dummy_hierarchy = core.Hierarchy(layers=[seg1, seg2])

    try:
        result = ops.boundary_salience(dummy_hierarchy)
        # The stub currently returns None
        assert result is None
    except Exception as e:
        pytest.fail(f"ops.boundary_salience raised an exception: {e}")

    # Test with the 'r' parameter
    try:
        result_with_r = ops.boundary_salience(dummy_hierarchy, r=1.5)
        assert result_with_r is None
    except Exception as e:
        pytest.fail(f"ops.boundary_salience with r param raised an exception: {e}")


# To run this test specifically:
# pixi run pytest tests/test_ops.py
