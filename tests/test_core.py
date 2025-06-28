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
    with pytest.raises(ValueError, match="Segments must be non-overlapping and contiguous."):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1), bnl.TimeSpan(2, 3)])

    # Test for overlapping segments
    with pytest.raises(ValueError, match="Segments must be non-overlapping and contiguous."):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1.5), bnl.TimeSpan(1, 2)])

    # Test for inconsistent layer durations in Hierarchy
    with pytest.raises(ValueError, match="All layers must have the same start and end time."):
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
    seg2 = bnl.Segmentation.from_intervals(np.array([[0.0, 0.5], [0.5, 1.0]]), ["B", "C"])
    hierarchy = bnl.Hierarchy(layers=[seg, seg2])
    assert str(hierarchy) == "Hierarchy(2 levels over 0.00s-1.00s)"
    assert repr(hierarchy) == "Hierarchy(2 levels over 0.00s-1.00s)"

    assert str(seg) == "Segmentation(1 segments over 1.00s)"
    assert repr(seg) == "Segmentation(1 segments over 1.00s)"
    assert str(seg2) == "Segmentation(2 segments over 1.00s)"
    assert repr(seg2) == "Segmentation(2 segments over 1.00s)"


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


def test_plotting_runs_without_error():
    """A lean test to ensure plotting functions can be called without error."""
    # Test TimeSpan plotting
    span_named = bnl.TimeSpan(start=0.0, end=1.0, name="test_span")
    fig, ax = span_named.plot()
    plt.close(fig)

    span_unnamed = bnl.TimeSpan(start=0.0, end=1.0)
    fig, ax = span_unnamed.plot(text=False)
    plt.close(fig)

    # Test Segmentation plotting
    seg = bnl.Segmentation.from_boundaries([0, 1, 2], ["X", "Y"])
    fig, ax = seg.plot()
    plt.close(fig)

    # Test Segmentation plotting with no segments
    empty_seg = bnl.Segmentation()
    fig, ax = empty_seg.plot()  # Should plot an empty axes
    plt.close(fig)

    # Test Hierarchy plotting
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"])
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 2.0], ["a", "b"])
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2])
    fig = hierarchy.plot()
    plt.close(fig)

    # Test Hierarchy plotting with one layer
    hierarchy_single_layer = bnl.Hierarchy(layers=[seg1])
    fig = hierarchy_single_layer.plot()
    plt.close(fig)

    # Test plotting with style_map
    fig, ax = span_named.plot(color="red", ymax=0.5)
    plt.close(fig)

    # Test plotting hierarchy with empty layers (should raise error)
    with pytest.raises(ValueError, match="Cannot plot empty hierarchy"):
        empty_hierarchy = bnl.Hierarchy()
        empty_hierarchy.plot()


def test_segmentation_from_jams(mocker):
    """Test creating a Segmentation from a JAMS annotation."""
    # Mock JAMS annotation
    mock_anno = mocker.MagicMock()
    mock_obs1 = mocker.MagicMock()
    mock_obs1.time = 0.0
    mock_obs1.duration = 1.0
    mock_obs1.value = "segment1"
    mock_obs2 = mocker.MagicMock()
    mock_obs2.time = 1.0
    mock_obs2.duration = 1.5
    mock_obs2.value = "segment2"
    mock_anno.__iter__.return_value = [mock_obs1, mock_obs2]

    seg = bnl.Segmentation.from_jams(mock_anno)
    assert len(seg) == 2
    assert seg.segments[0].start == 0.0
    assert seg.segments[0].end == 1.0
    assert seg.segments[0].name == "segment1"
    assert seg.segments[1].start == 1.0
    assert seg.segments[1].end == 2.5
    assert seg.segments[1].name == "segment2"


def test_hierarchy_from_jams(mocker):
    """Test creating a Hierarchy from a JAMS multi_segment annotation."""
    # Mock JAMS annotation and hierarchy_flatten
    mock_anno = mocker.MagicMock()
    mock_anno.namespace = "multi_segment"

    # Mock return value of hierarchy_flatten
    # Represents two levels:
    # Level 0: [(0.0, 5.0, "A")]
    # Level 1: [(0.0, 2.0, "a"), (2.0, 5.0, "b")]
    mock_hier_intervals = [[(0.0, 5.0)], [(0.0, 2.0), (2.0, 5.0)]]
    mock_hier_labels = [["A"], ["a", "b"]]
    mocker.patch("jams.eval.hierarchy_flatten", return_value=(mock_hier_intervals, mock_hier_labels))

    hierarchy = bnl.Hierarchy.from_jams(mock_anno)
    assert len(hierarchy) == 2
    assert len(hierarchy.layers[0]) == 1
    assert hierarchy.layers[0].segments[0].name == "A"
    assert hierarchy.layers[0].segments[0].start == 0.0
    assert hierarchy.layers[0].segments[0].end == 5.0
    assert len(hierarchy.layers[1]) == 2
    assert hierarchy.layers[1].segments[0].name == "a"
    assert hierarchy.layers[1].segments[1].name == "b"

    # Test wrong namespace
    mock_anno.namespace = "wrong_namespace"
    with pytest.raises(ValueError, match="Expected 'multi_segment' namespace"):
        bnl.Hierarchy.from_jams(mock_anno)


def test_hierarchy_not_implemented_constructors():
    """Test that from_json, from_boundaries, from_intervals raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        bnl.Hierarchy.from_json({})
    with pytest.raises(NotImplementedError):
        bnl.Hierarchy.from_boundaries([[]])
    with pytest.raises(NotImplementedError):
        bnl.Hierarchy.from_intervals([np.array([])])
