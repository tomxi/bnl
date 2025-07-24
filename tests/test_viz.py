"""Tests for the visualization module."""

import plotly.graph_objects as go
import pytest

from bnl import core, viz


@pytest.fixture
def sample_multisegment() -> core.MultiSegment:
    """Returns a sample MultiSegment for testing."""
    s1 = core.Segment.from_bs([0, 1, 2, 3], ["A", "B", ""], name="L1")
    s2 = core.Segment.from_bs([0, 1.5, 3], ["C", ""], name="L2")
    return core.MultiSegment(layers=[s1, s2], name="Sample MS")


@pytest.fixture
def sample_boundary_contour() -> core.BoundaryContour:
    """Returns a sample BoundaryContour for testing."""
    return core.BoundaryContour(
        bs=[
            core.RatedBoundary(0, 1.0),
            core.RatedBoundary(1, 0.5),
            core.RatedBoundary(2, 0.8),
            core.RatedBoundary(3, 1.0),
        ]
    )


def test_plot_multisegment(sample_multisegment: core.MultiSegment):
    """Test that plot_multisegment runs and returns a Figure."""
    fig = viz.plot_multisegment(sample_multisegment)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Sample MS"


def test_plot_boundary_contour(sample_boundary_contour: core.BoundaryContour):
    """Test that plot_boundary_contour runs and returns a Figure."""
    fig = viz.plot_boundary_contour(sample_boundary_contour)
    assert isinstance(fig, go.Figure)
    assert "BoundaryContour" in fig.layout.title.text


def test_create_style_map():
    """Test the create_style_map function."""
    labels = ["A", "B", "C"]
    style_map = viz.create_style_map(labels)
    assert isinstance(style_map, dict)
    assert "A" in style_map
    assert "color" in style_map["A"]
    assert "pattern_shape" in style_map["A"]
