import matplotlib.pyplot as plt
import numpy as np
import pytest

import bnl


def test_segmentation_basic_init():
    """Test basic segmentation functionality."""
    seg = bnl.Segmentation(
        segments=[
            bnl.TimeSpan(start=0.0, end=1.5, name="verse"),
            bnl.TimeSpan(start=1.5, end=3.0, name="chorus"),
        ]
    )
    assert seg.bdrys == [0.0, 1.5, 3.0]
    assert seg.labels == ["verse", "chorus"]
    assert len(seg) == 2


def test_hierarchy_basic():
    """Test basic hierarchy functionality."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"])
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 2.0], ["a", "b"])
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2])
    assert len(hierarchy) == 2
    assert hierarchy[0] == seg1


def test_post_init_errors():
    """Test that __post_init__ raises errors for malformed objects."""
    # Test for non-contiguous segments
    with pytest.raises(
        ValueError, match="Segments must be non-overlapping and contiguous."
    ):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1), bnl.TimeSpan(2, 3)])

    # Test for overlapping segments
    with pytest.raises(
        ValueError, match="Segments must be non-overlapping and contiguous."
    ):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1.5), bnl.TimeSpan(1, 2)])

    # Test for inconsistent layer durations in Hierarchy
    with pytest.raises(
        ValueError, match="All layers must have the same start and end time."
    ):
        seg1 = bnl.Segmentation.from_boundaries([0, 2])
        seg2 = bnl.Segmentation.from_boundaries([0, 1, 3])
        bnl.Hierarchy(layers=[seg1, seg2])


@pytest.mark.parametrize(
    "constructor, data",
    [
        (bnl.Segmentation.from_intervals, np.array([[0.0, 1.0], [1.0, 2.5]])),
        (bnl.Segmentation.from_boundaries, [0.0, 1.0, 2.5]),
    ],
)
def test_segmentation_constructors(constructor, data):
    """Test Segmentation constructors with and without labels and names."""
    # Test with labels and a name
    seg1 = constructor(data, labels=["A", "B"], name="TestName")
    assert seg1.labels == ["A", "B"]
    assert seg1.name == "TestName"
    np.testing.assert_array_equal(seg1.itvls, np.array([[0.0, 1.0], [1.0, 2.5]]))

    # Test without labels (default labels)
    seg2 = constructor(data)
    assert seg2.labels == ["[0.0-1.0s]", "[1.0-2.5s]"]
    assert seg2[0] == bnl.TimeSpan(start=0.0, end=1.0, name="[0.0-1.0s]")


def test_str_repr():
    """Test string representation of core classes."""
    seg = bnl.Segmentation(segments=[bnl.TimeSpan(start=0.0, end=1.0, name="A")])
    seg2 = bnl.Segmentation.from_intervals(
        np.array([[0.0, 0.5], [0.5, 1.0]]), ["B", "C"]
    )
    hierarchy = bnl.Hierarchy(layers=[seg, seg2])
    assert str(hierarchy) == "Hierarchy(2 levels over 0.00s-1.00s)"
    assert repr(hierarchy) == "Hierarchy(2 levels over 0.00s-1.00s)"

    assert str(seg) == "Segmentation(1 segments over 1.00s)"
    assert repr(seg) == "Segmentation(1 segments over 1.00s)"
    assert str(seg2) == "Segmentation(2 segments over 1.00s)"
    assert repr(seg2) == "Segmentation(2 segments over 1.00s)"


def test_unimplemented_methods():
    """Cover unimplemented methods for test coverage."""
    bnl.Hierarchy.from_jams(None)
    bnl.Segmentation.from_jams(None)
    bnl.Hierarchy(layers=[]).plot()


def test_core_edge_cases_and_validation():
    """Test edge cases and validation for core classes."""
    # Test TimeSpan validation error
    with pytest.raises(ValueError):
        bnl.TimeSpan(start=2.0, end=1.0)

    # Test empty segmentation cases
    empty_seg = bnl.Segmentation()
    assert empty_seg.itvls.size == 0
    assert empty_seg.bdrys == []
    assert str(empty_seg) == "Segmentation(0 segments): []"

    # Test empty hierarchy cases
    empty_hierarchy = bnl.Hierarchy()
    assert str(empty_hierarchy) == "Hierarchy(0 levels)"
    assert empty_hierarchy.itvls == []
    assert empty_hierarchy.labels == []
    assert empty_hierarchy.bdrys == []

    # Test TimeSpan without name
    unnamed_span = bnl.TimeSpan(start=1.0, end=2.0)
    assert str(unnamed_span) == "[1.0-2.0s][1.0-2.0s]"
    assert repr(unnamed_span) == "TimeSpan([1.0-2.0s][1.0-2.0s])"


def test_timespan_plot_full_coverage():
    """Test TimeSpan.plot method for full branch coverage."""
    # Test with a named span to cover default path for text and color handling
    span_named = bnl.TimeSpan(start=0.0, end=1.0, name="test_span")

    # Cover `if "color" in style_map:` branch and basic text
    fig1, ax1 = span_named.plot(color="blue")
    assert ax1.patches[0].get_facecolor() == (0.0, 0.0, 1.0, 1.0)
    assert ax1.texts[0].get_text() == "test_span"  # Ensure text is plotted
    plt.close(fig1)

    # Cover `if text:` branch when text is False
    fig2, ax2 = span_named.plot(text=False)
    assert len(ax2.texts) == 0
    plt.close(fig2)

    # Cover `style_map.setdefault("edgecolor", "white")` when no style is passed (and default text)
    fig3, ax3 = span_named.plot()
    assert ax3.patches[0].get_edgecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax3.texts[0].get_text() == "test_span"  # Ensure text is plotted
    plt.close(fig3)

    # Test with an unnamed span to cover `lab = self.name if self.name else str(self)`'s else branch
    span_unnamed = bnl.TimeSpan(start=0.0, end=1.0)  # name=None
    fig4, ax4 = span_unnamed.plot()  # text=True by default
    assert ax4.texts[0].get_text() == "[0.0-1.0s]"  # Expect str(self)
    plt.close(fig4)
