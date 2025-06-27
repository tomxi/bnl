import librosa.display
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
    # Basic functionality
    labels = ["A", "B", "A", "C"]
    styles = viz.label_style_dict(labels)
    assert isinstance(styles, dict)
    assert set(styles.keys()) == {"A", "B", "C"}

    # Essential properties exist
    for label in ["A", "B", "C"]:
        assert all(key in styles[label] for key in ["facecolor", "edgecolor", "label"])

    # Custom boundary color
    styles = viz.label_style_dict(["X"], boundary_color="blue")
    assert styles["X"]["edgecolor"] == "blue"

    # Many labels (>80) - test different style generation
    many_labels = [f"label_{i}" for i in range(81)]
    styles = viz.label_style_dict(many_labels)
    assert len(styles) == 81


def test_segmentation_plotting():
    """Test segmentation plotting functionality."""
    seg = Segmentation.from_boundaries([0, 1, 2], ["X", "Y"])

    # Basic plotting
    fig, ax = seg.plot(text=True, ytick="Test")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.texts) == 2  # Text labels
    assert ax.get_yticklabels()[0].get_text() == "Test"

    # Time formatting
    fig, ax = viz.plot_segment(seg, time_ticks=True)
    assert isinstance(ax.xaxis.get_major_formatter(), librosa.display.TimeFormatter)

    fig, ax = viz.plot_segment(seg, time_ticks=False)
    assert len(ax.get_xticks()) == 0

    # Edge cases - single segment
    single_seg = Segmentation.from_boundaries([0, 1], ["X"])
    fig, ax = viz.plot_segment(single_seg)
    assert fig is not None


def test_empty_segmentation_with_title_and_ytick():
    """Test plotting empty segmentation with title and ytick.

    This test is designed to cover specific lines in the plotting function
    (lines 162, 172, 184 in the original implementation).
    """
    # Create empty segmentation with a name
    empty_seg = Segmentation(name="empty")

    # Create existing axis
    fig, ax = plt.subplots()

    # Plot with title and ytick to cover all missed lines
    fig2, ax2 = viz.plot_segment(empty_seg, ax=ax, title=True, ytick="Empty Level")

    # Verify coverage: line 162 - "Empty Segmentation" text
    texts = [t.get_text() for t in ax.texts]
    assert "Empty Segmentation" in texts

    # Verify coverage: line 172 - title is set when seg.name exists
    assert ax.get_title() == "empty"

    # Verify coverage: line 184 - ytick labels are set
    assert ax.get_yticklabels()[0].get_text() == "Empty Level"
