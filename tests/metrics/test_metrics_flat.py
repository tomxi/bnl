"""Tests for flat segmentation metrics."""

import pytest

from bnl import metrics # or bnl.metrics.flat directly


def test_f_measure_stub():
    """Test that the f_measure stub can be called."""
    # Since it's a stub, we can't test functionality yet.
    # We can test if it's callable and doesn't raise an error for dummy inputs.
    ref_dummy = [1.0, 2.0, 3.0] # Dummy reference boundaries
    est_dummy = [1.1, 2.5]    # Dummy estimated boundaries

    try:
        result = metrics.f_measure(ref_dummy, est_dummy)
        # Stubs currently 'pass', which means they return None implicitly
        assert result is None
    except Exception as e:
        pytest.fail(f"f_measure stub raised an exception: {e}")

    # Test with window argument
    try:
        result_window = metrics.f_measure(ref_dummy, est_dummy, window=1.0)
        assert result_window is None
    except Exception as e:
        pytest.fail(f"f_measure stub with window raised an exception: {e}")

# To run this test specifically:
# pixi run pytest tests/metrics/test_metrics_flat.py
