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
    fig, _ = hierarchy.plot_single_axis()  # Changed to plot_single_axis
    plt.close(fig)

    # Test Hierarchy plotting with one layer
    hierarchy_single_layer = bnl.Hierarchy(layers=[seg1])
    fig, _ = hierarchy_single_layer.plot_single_axis()  # Changed to plot_single_axis
    plt.close(fig)

    # Test plotting with style_map
    fig, ax = span_named.plot(color="red", ymax=0.5)
    plt.close(fig)

    # Test plotting hierarchy with empty layers (should raise error for plot_single_axis)
    with pytest.raises(ValueError, match="Cannot plot empty hierarchy"):
        empty_hierarchy = bnl.Hierarchy()
        empty_hierarchy.plot_single_axis()


@pytest.mark.xfail(reason="from_jams methods are not implemented in this task")
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


@pytest.mark.xfail(reason="from_jams methods are not implemented in this task")
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
    """Test that from_boundaries, from_intervals raise NotImplementedError."""
    # bnl.Hierarchy.from_json is now implemented.
    with pytest.raises(NotImplementedError):
        bnl.Hierarchy.from_boundaries([[]])
    with pytest.raises(NotImplementedError):
        bnl.Hierarchy.from_intervals([np.array([])])


import json  # For parsing json

import requests  # For fetching real annotation file


def test_hierarchy_from_json_adobe_est_format():
    """Test creating a Hierarchy from the Adobe EST JSON structure."""
    # Structure: list of layers, each layer is [intervals_list, labels_list]
    # where intervals_list is list of [start, end] or list of [[start, end]]
    json_data_valid = [
        [[[0.0, 10.0]], ["Layer0_SegA"]],  # Layer 0: 1 segment
        [[[0.0, 5.0], [5.0, 10.0]], ["Layer1_Sega", "Layer1_Segb"]],  # Layer 1: 2 segments
        [
            [[0.0, 2.5], [2.5, 5.0], [5.0, 7.5], [7.5, 10.0]],
            ["L2_s1", "L2_s2", "L2_s3", "L2_s4"],
        ],  # Layer 2: 4 segments
    ]
    hierarchy = bnl.Hierarchy.from_json(json_data_valid, name="TestAdobeESTHierarchy")

    assert hierarchy.name == "TestAdobeESTHierarchy"
    assert len(hierarchy) == 3
    assert len(hierarchy.layers[0]) == 1
    assert hierarchy.layers[0].segments[0].name == "Layer0_SegA"
    assert hierarchy.layers[0].segments[0].start == 0.0
    assert hierarchy.layers[0].segments[0].end == 10.0

    assert len(hierarchy.layers[1]) == 2
    assert hierarchy.layers[1].segments[1].name == "Layer1_Segb"
    assert hierarchy.layers[1].segments[1].start == 5.0

    assert len(hierarchy.layers[2]) == 4
    assert hierarchy.layers[2].segments[3].name == "L2_s4"
    assert hierarchy.layers[2].segments[3].end == 10.0

    # Test with the more complex interval structure [[start, end]] per segment
    json_data_complex_interval = [
        [[[0.0, 10.0]], ["Layer0_SegA"]],
    ]
    hierarchy_complex_interval = bnl.Hierarchy.from_json(json_data_complex_interval)
    assert len(hierarchy_complex_interval.layers[0]) == 1
    assert hierarchy_complex_interval.layers[0].segments[0].start == 0.0
    assert hierarchy_complex_interval.layers[0].segments[0].end == 10.0

    # Test with empty list (no layers)
    empty_hierarchy = bnl.Hierarchy.from_json([])
    assert len(empty_hierarchy) == 0
    assert empty_hierarchy.name is None

    # Test with a layer that has no segments (empty intervals_list and labels_list)
    json_data_empty_layer = [
        [[[0.0, 10.0]], ["Layer0_SegA"]],
        [[], []],  # Empty layer
        [[[0.0, 10.0]], ["Layer2_SegA"]],
    ]
    hierarchy_empty_layer = bnl.Hierarchy.from_json(json_data_empty_layer)
    assert len(hierarchy_empty_layer) == 3
    assert len(hierarchy_empty_layer.layers[0]) == 1
    assert len(hierarchy_empty_layer.layers[1]) == 0
    assert len(hierarchy_empty_layer.layers[2]) == 1
    assert hierarchy_empty_layer.start == 0.0
    assert hierarchy_empty_layer.end == 10.0
    assert hierarchy_empty_layer.layers[1].start == 0.0
    assert hierarchy_empty_layer.layers[1].end == 0.0

    # Test malformed layer (not a list of two lists)
    json_data_malformed_layer = [[[[0.0, 10.0]], ["A"]], "not a layer"]
    with pytest.raises(ValueError, match="Layer 1 is malformed"):
        bnl.Hierarchy.from_json(json_data_malformed_layer)

    # Test mismatched intervals and labels
    json_data_mismatched = [[[[0.0, 5.0], [5.0, 10.0]], ["A"]]]
    with pytest.raises(ValueError, match="Layer 0 has mismatched number of intervals and labels"):
        bnl.Hierarchy.from_json(json_data_mismatched)

    # Test malformed interval item
    json_data_malformed_interval = [[[[0.0, 5.0, 6.0]], ["A"]]]
    with pytest.raises(ValueError, match="Malformed interval structure in layer 0"):
        bnl.Hierarchy.from_json(json_data_malformed_interval)

    json_data_malformed_interval_2 = [[[["string", 5.0]], ["A"]]]
    with pytest.raises(ValueError, match="could not convert string to float"):
        bnl.Hierarchy.from_json(json_data_malformed_interval_2)

    # Test inconsistent layer durations
    json_data_inconsistent_duration = [
        [[[0.0, 10.0]], ["A"]],
        [[[0.0, 12.0]], ["b"]],
    ]
    with pytest.raises(ValueError, match="All layers must have the same start and end time."):
        bnl.Hierarchy.from_json(json_data_inconsistent_duration)


@pytest.mark.remote_data
def test_hierarchy_from_json_real_file():
    """Test parsing a real Adobe EST JSON file from the R2 bucket."""
    annotation_url = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/adobe21-est/def_mu_0.1_gamma_0.1/10.mp3.msdclasscsnmagic.json"

    try:
        response = requests.get(annotation_url, timeout=10)
        response.raise_for_status()
        real_json_data = response.json()

        hierarchy = bnl.Hierarchy.from_json(real_json_data, name="FetchedRealData")

        assert hierarchy is not None
        assert hierarchy.name == "FetchedRealData"
        assert len(hierarchy) > 0

        if len(hierarchy) > 0 and len(hierarchy.layers[0]) > 0:
            first_segment = hierarchy.layers[0].segments[0]
            assert isinstance(first_segment.start, float)
            assert isinstance(first_segment.end, float)
            assert first_segment.start <= first_segment.end
            assert isinstance(first_segment.name, str)

        assert isinstance(hierarchy.start, float)
        assert isinstance(hierarchy.end, float)
        if len(hierarchy) > 0:
            assert hierarchy.start <= hierarchy.end

    except requests.exceptions.RequestException as e:
        pytest.skip(f"Skipping remote data test, could not fetch {annotation_url}: {e}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to decode JSON from {annotation_url}: {e}")
    except ValueError as e:
        pytest.fail(f"ValueError during Hierarchy parsing from {annotation_url}: {e}")


def test_hierarchy_plot_single_axis():
    """Test the plot_single_axis method for Hierarchy."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0, 4.0], ["A1", "A2"], name="L0")
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 2.0, 3.0, 4.0], ["b1", "b2", "b3", "b4"], name="L1")
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2], name="SingleAxisTest")

    fig, ax = hierarchy.plot_single_axis()
    assert fig is not None
    assert ax is not None
    assert len(fig.axes) == 1

    expected_yticks = ["Level 1", "Level 0"]
    actual_yticks = [label.get_text() for label in ax.get_yticklabels()]
    assert actual_yticks == expected_yticks

    plt.close(fig)

    seg_empty = bnl.Segmentation(start=0.0, end=4.0)
    hierarchy_with_empty = bnl.Hierarchy(layers=[seg1, seg_empty, seg2])
    fig_empty, ax_empty = hierarchy_with_empty.plot_single_axis()
    assert len(ax_empty.get_yticklabels()) == 3
    expected_yticks_empty = ["Level 2", "Level 1", "Level 0"]
    actual_yticks_empty = [label.get_text() for label in ax_empty.get_yticklabels()]
    assert actual_yticks_empty == expected_yticks_empty

    plt.close(fig_empty)

    with pytest.raises(ValueError, match="Cannot plot empty hierarchy"):
        empty_hierarchy = bnl.Hierarchy()
        empty_hierarchy.plot_single_axis()
