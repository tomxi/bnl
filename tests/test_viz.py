import matplotlib.pyplot as plt
import pytest

from bnl import Segmentation, viz


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib plots after each test."""
    yield
    plt.close("all")


def test_label_style_dict():
    """Test style generation for labels."""
    labels = ["A", "B", "A", "C"]
    styles = viz.label_style_dict(labels)
    assert isinstance(styles, dict)
    assert set(styles.keys()) == {"A", "B", "C"}
    assert "facecolor" in styles["A"]


def test_segmentation_plotting_runs_without_error():
    """Test that core visualization functions run without error."""
    # Test with a standard segmentation
    seg = Segmentation.from_boundaries([0, 1, 2], ["X", "Y"], name="TestSeg")
    fig, ax = viz.plot_segment(seg, label_text=True, ytick="Test", time_ticks=True, title=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test with an empty segmentation
    empty_seg = Segmentation(name="EmptySeg")
    fig, ax = viz.plot_segment(empty_seg, title=True)
    assert "Empty Segmentation" in [t.get_text() for t in ax.texts]
    assert ax.get_title() == "EmptySeg"
