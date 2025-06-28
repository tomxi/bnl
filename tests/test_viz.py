import matplotlib.pyplot as plt
import pytest

from bnl import Segmentation, viz, TimeSpan # Added TimeSpan


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

    # Test with more than 80 unique labels to hit the other cycler branch
    many_labels = [f"Label_{i}" for i in range(85)]
    many_styles = viz.label_style_dict(many_labels)
    assert isinstance(many_styles, dict)
    assert len(many_styles.keys()) == 85
    assert "facecolor" in many_styles["Label_0"]
    # Check that it uses tab20 colormap (or at least more colors)
    # This is an indirect check; direct check of cmap is harder.
    # If it used tab10, colors would repeat much sooner.
    # Here, we just ensure it runs and produces the expected structure.
    assert many_styles["Label_0"]["facecolor"] != many_styles["Label_15"]["facecolor"] # tab20 has 20 distinct colors


def test_segmentation_plotting_runs_without_error():
    """Test that core visualization functions run without error."""
    # Test with a standard segmentation
    seg = Segmentation.from_boundaries([0, 1, 2], ["X", "Y"], name="TestSeg")
    fig, ax = viz.plot_segment(seg, label_text=True, ytick="Test", time_ticks=True, title=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test with an empty segmentation
    empty_seg = Segmentation(name="EmptySeg") # start=0, end=0 by default
    fig_empty, ax_empty = viz.plot_segment(empty_seg, title=True)
    assert "Empty Segmentation" in [t.get_text() for t in ax_empty.texts]
    assert ax_empty.get_title() == "EmptySeg"
    assert ax_empty.get_xlim() == (-0.1, 0.1) # Covers seg.start == seg.end

    # Test with time_ticks=False (covers line 151)
    fig_no_ticks, ax_no_ticks = viz.plot_segment(seg, time_ticks=False)
    assert len(ax_no_ticks.get_xticks()) == 0

    # Test with seg.name = None (covers conditional title)
    seg_no_name = Segmentation.from_boundaries([0, 1], ["X"]) # name will be None
    fig_no_name, ax_no_name = viz.plot_segment(seg_no_name, title=True)
    assert ax_no_name.get_title() == "" # No title should be set

    # Test for span.name not in style_map or span.name is None (covers line 108 default get)
    # Create a segmentation where one span has a name that won't be in the auto-style_map
    # if style_map is generated only from other labels, or a span with name=None.
    span_named = TimeSpan(0, 1, "Known")
    span_none_name = TimeSpan(1, 2, None)
    span_unknown_name = TimeSpan(2, 3, "UnknownInMap")

    # Case 1: Explicit style_map that misses "UnknownInMap" and handles "" for None
    seg_mixed_names = Segmentation(segments=[span_named, span_none_name, span_unknown_name])
    custom_style_map = viz.label_style_dict(["Known"]) # Only "Known" is in map
    custom_style_map[""] = {"facecolor": "gray"} # Style for None name (becomes "" key)

    fig_mixed, ax_mixed = viz.plot_segment(seg_mixed_names, style_map=custom_style_map)
    # This ensures that get(span.name or "", {}) was called and didn't crash.
    # Further checks could inspect the actual colors if specific defaults were expected.
    assert len(ax_mixed.patches) == 3 # Check that all spans were plotted

    # Case 2: Auto-generated style_map with a None name span
    # The auto-style map generated from `seg_mixed_names.labels` (["Known", None, "UnknownInMap"])
    # will handle None correctly if None becomes a key or is filtered out.
    # If span.name is None, key becomes "" for style_map.get.
    # label_style_dict filters out None labels before creating styles.
    # So, a span with name=None will result in style_map.get("", {})
    seg_with_none_span = Segmentation(segments=[span_named, span_none_name])
    fig_none_span, ax_none_span = viz.plot_segment(seg_with_none_span) # Auto style map
    assert len(ax_none_span.patches) == 2
