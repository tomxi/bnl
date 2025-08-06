"""Tests for the visualization module."""

import plotly.graph_objects as go
import pytest

import bnl
from bnl import viz


@pytest.fixture
def sample_multisegment() -> bnl.MS:
    """Returns a sample MultiSegment for testing."""
    s1 = bnl.S.from_bs([0, 1, 2, 3], ["A", "B", ""], name="L1")
    s2 = bnl.S.from_bs([0, 1.5, 3], ["C", ""], name="L2")
    return bnl.MS(raw_layers=[s1, s2], name="Sample MS")


@pytest.fixture
def sample_boundary_contour() -> bnl.BC:
    """Returns a sample BoundaryContour for testing."""
    return bnl.BC(
        bs=[
            bnl.RB(0, 1.0),
            bnl.RB(1, 0.5),
            bnl.RB(2, 0.8),
            bnl.RB(3, 1.0),
        ]
    )


def test_plot_multisegment(sample_multisegment: bnl.MS):
    """Test that plot_multisegment runs and returns a Figure."""
    fig = viz.plot_multisegment(sample_multisegment)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Sample MS"


def test_plot_boundary_contour(sample_boundary_contour: bnl.BC):
    """Test that plot_boundary_contour runs and returns a Figure."""
    fig = viz.plot_boundary_contour(sample_boundary_contour)
    assert isinstance(fig, go.Figure)
    assert "BC" in fig.layout.title.text


def test_create_style_map():
    """Test the create_style_map function."""
    labels = ["A", "B", "C"]
    style_map = viz.create_style_map(labels)
    assert isinstance(style_map, dict)
    assert "A" in style_map
    assert "color" in style_map["A"]
    assert "pattern_shape" in style_map["A"]
