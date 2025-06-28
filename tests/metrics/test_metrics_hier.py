"""Tests for hierarchical segmentation metrics."""

import pytest

from bnl import metrics  # or bnl.metrics.hier directly

# t_measure is not exposed via bnl.metrics, so import directly if needed for future
from bnl.metrics.hier import t_measure


def test_l_measure_stub():
    """Test that the l_measure stub can be called."""
    ref_dummy_hier = [[1.0, 5.0], [1.0, 2.5, 5.0]]  # Dummy hierarchical reference
    est_dummy_hier = [[0.8, 4.5], [0.8, 2.0, 3.0, 4.5]]  # Dummy hierarchical estimation

    try:
        result = metrics.l_measure(ref_dummy_hier, est_dummy_hier)
        assert result is None  # Stubs return None
    except Exception as e:
        pytest.fail(f"l_measure stub raised an exception: {e}")

    # Test with window argument
    try:
        result_window = metrics.l_measure(ref_dummy_hier, est_dummy_hier, window=1.0)
        assert result_window is None
    except Exception as e:
        pytest.fail(f"l_measure stub with window raised an exception: {e}")


def test_t_measure_stub():
    """Test that the t_measure stub can be called."""
    # t_measure is not in bnl.metrics top-level, so call it directly from bnl.metrics.hier
    ref_dummy_hier = [[1.0, 5.0], [1.0, 2.5, 5.0]]
    est_dummy_hier = [[0.8, 4.5], [0.8, 2.0, 3.0, 4.5]]

    try:
        # Note: Using t_measure directly from its module for this test
        result = t_measure(ref_dummy_hier, est_dummy_hier)
        assert result is None  # Stubs return None
    except Exception as e:
        pytest.fail(f"t_measure stub raised an exception: {e}")

    try:
        result_window = t_measure(ref_dummy_hier, est_dummy_hier, window=1.0)
        assert result_window is None
    except Exception as e:
        pytest.fail(f"t_measure stub with window raised an exception: {e}")


# To run this test specifically:
# pixi run pytest tests/metrics/test_metrics_hier.py
