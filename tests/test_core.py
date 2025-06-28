import matplotlib.pyplot as plt
import numpy as np
import pytest
import jams
from pathlib import Path
import json
import requests

import bnl


def test_segmentation_basic_init():
    """Test basic segmentation functionality."""
    seg = bnl.Segmentation(
        segments=[
            bnl.TimeSpan(start=0.0, end=1.5, name="verse"),
            bnl.TimeSpan(start=1.5, end=3.0, name="chorus"),
        ],
        name="song_structure",
    )
    assert seg.boundaries == [0.0, 1.5, 3.0]
    assert seg.labels == ["verse", "chorus"]
    assert len(seg) == 2
    assert seg.name == "song_structure"
    assert seg.duration == 3.0


def test_hierarchy_basic():
    """Test basic hierarchy functionality."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"], name="Coarse")
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 2.0], ["a", "b"], name="Fine")
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2], name="MyHierarchy")
    assert len(hierarchy) == 2
    assert hierarchy[0] == seg1
    assert hierarchy.name == "MyHierarchy"
    assert hierarchy.duration == 2.0
    assert hierarchy.layers[0].name == "Coarse"


def test_post_init_errors():
    """Test that __post_init__ raises errors for malformed objects."""
    # Test TimeSpan errors
    with pytest.raises(ValueError, match="Start time .* cannot be negative"):
        bnl.TimeSpan(start=-1.0, end=1.0)
    with pytest.raises(ValueError, match="End time .* cannot be negative"):
        bnl.TimeSpan(start=0.0, end=-1.0)
    with pytest.raises(ValueError, match="Start time .* must be less than or equal to end time"):
        bnl.TimeSpan(start=2.0, end=1.0)

    # Test Segmentation errors for non-event types
    with pytest.raises(ValueError, match="Segments must be contiguous for this segmentation type."):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1), bnl.TimeSpan(2, 3)], name="structural_segments")

    # Test for segments that are non-contiguous AND overlapping. Contiguity error should be raised first.
    with pytest.raises(ValueError, match="Segments must be contiguous for this segmentation type."):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1.5), bnl.TimeSpan(1.0, 2.0)], name="structural_segments_noncontig_overlap")

    # Test for segments that are non-contiguous (gap after a valid segment). Contiguity error should be raised.
    with pytest.raises(ValueError, match="Segments must be contiguous for this segmentation type."):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1.5), bnl.TimeSpan(1.5, 2.5), bnl.TimeSpan(2.0, 3.0)], name="structural_segments_gap")

    # To properly test the "non-overlapping" error as currently implemented,
    # segments must be non-contiguous in a way that self.segments[i].end > self.segments[i+1].start
    # AND np.isclose(self.segments[i].end, self.segments[i+1].start) is false.
    # Example: seg1=(0, 2.0), seg2=(1.99, 3.0)
    # Here, 2.0 > 1.99 is true. np.isclose(2.0, 1.99, atol=1e-9) is false.
    # This case should trigger the "non-overlapping" error *if* the contiguity check was designed differently
    # or if the overlap check was primary.
    # With current logic, this input would also fail contiguity first:
    # isclose(2.0, 1.99) is false.
    # The current "non-overlapping" check might be difficult to isolate.
    # For now, we'll rely on the contiguity check catching most malformations for non-event types.

    # Test for inconsistent layer durations in Hierarchy
    with pytest.raises(ValueError, match="All non-empty layers in a Hierarchy must span the same overall time range."):
        seg_h1 = bnl.Segmentation.from_boundaries([0, 2], name="s1")
        seg_h2 = bnl.Segmentation.from_boundaries([0, 1, 3], name="s2") # Different end time
        bnl.Hierarchy(layers=[seg_h1, seg_h2])

    # Test Hierarchy with one valid layer and one empty layer (should be fine) - This test is no longer valid
    # as Segmentation cannot be empty, and Hierarchy layers are Segmentations.
    # seg_valid = bnl.Segmentation.from_boundaries([0,2], name="valid_layer")
    # # seg_empty = bnl.Segmentation(name="empty_layer") # This line would now raise ValueError
    # # try:
    # #     h = bnl.Hierarchy(layers=[seg_valid, seg_empty]) # This would fail if seg_empty could be created
    # #     assert h.start == 0.0 and h.end == 2.0
    # # except ValueError:
    # #     pytest.fail("Hierarchy with one valid and one empty layer should not raise ValueError on init.")

    # Test Hierarchy with layers of different start/end after some are empty
    # This test also becomes more complex as empty segmentations are not allowed.
    # The scenario of a hierarchy having "empty segmentations" as layers that affect overall duration calc
    # is no longer possible. The existing test for inconsistent layer durations for *non-empty* layers
    # (above) covers the primary validation.
    # The following test is also invalid as seg_b_empty cannot be created.
    # seg_a = bnl.Segmentation.from_intervals(np.array([[0.0, 5.0]]), name="A")
    # # seg_b_empty = bnl.Segmentation(name="B_empty") # This would raise ValueError
    # seg_c = bnl.Segmentation.from_intervals(np.array([[1.0, 6.0]]), name="C")
    # with pytest.raises(ValueError, match="All non-empty layers in a Hierarchy must span the same overall time range."):
    #     bnl.Hierarchy(layers=[seg_a, seg_b_empty, seg_c])


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
    np.testing.assert_array_equal(seg1.intervals, np.array([[0.0, 1.0], [1.0, 2.5]]))

    # Test without labels (default labels should be None)
    seg2 = constructor(data, name="NoLabelsSeg") # Name the segmentation itself
    assert seg2.labels == [None, None]
    assert seg2[0].name is None # TimeSpan name should be None
    assert seg2.name == "NoLabelsSeg" # Segmentation name
    assert seg2[0] == bnl.TimeSpan(start=0.0, end=1.0, name=None)


def test_str_repr():
    """Test string representation of core classes."""
    ts = bnl.TimeSpan(start=0.0, end=1.0, name="A")
    assert str(ts) == "TimeSpan([0.00s-1.00s], 1.00s: A)"
    assert repr(ts) == "TimeSpan(start=0.0, end=1.0, name='A')"

    ts_no_name = bnl.TimeSpan(start=0.5, end=1.5)
    assert str(ts_no_name) == "TimeSpan([0.50s-1.50s], 1.00s)" # No ": name" part
    assert repr(ts_no_name) == "TimeSpan(start=0.5, end=1.5, name='None')" # Explicit None for name in repr

    seg = bnl.Segmentation(segments=[ts], name="MySeg")
    seg_no_name = bnl.Segmentation(segments=[ts])
    seg2 = bnl.Segmentation.from_intervals(np.array([[0.0, 0.5], [0.5, 1.0]]), ["B", "C"], name="MySeg2")

    assert str(seg) == "Segmentation(name='MySeg', 1 segments, duration=1.00s)"
    assert repr(seg) == "Segmentation(name='MySeg', 1 segments, duration=1.00s)"
    assert str(seg_no_name) == "Segmentation(1 segments, duration=1.00s)" # No name part
    assert repr(seg_no_name) == "Segmentation(1 segments, duration=1.00s)" # No name part

    hierarchy = bnl.Hierarchy(layers=[seg, seg2], name="MyHier")
    hierarchy_no_name = bnl.Hierarchy(layers=[seg, seg2])

    assert str(hierarchy) == "Hierarchy(name='MyHier', 2 layers, duration=1.00s)"
    assert repr(hierarchy) == "Hierarchy(name='MyHier', 2 layers, duration=1.00s)"
    assert str(hierarchy_no_name) == "Hierarchy(name='None', 2 layers, duration=1.00s)" # name='None' if not provided
    assert repr(hierarchy_no_name) == "Hierarchy(name='None', 2 layers, duration=1.00s)"


def test_core_edge_cases_and_validation():
    """Test edge cases and validation for core classes."""
    # TimeSpan validation errors are covered in test_post_init_errors

    # Test empty segmentation cases - Now raises ValueError
    with pytest.raises(ValueError, match="Segmentation must contain at least one segment."):
        bnl.Segmentation(name="EmptyTestSeg") # Was: empty_seg = ...
    with pytest.raises(ValueError, match="Segmentation must contain at least one segment."):
        bnl.Segmentation() # Test case for init without name

    # Test empty hierarchy cases - Now raises ValueError
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        bnl.Hierarchy(name="EmptyTestHier") # Was: empty_hierarchy = ...
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        bnl.Hierarchy() # Test case for init without name

    # TimeSpan without name (already tested in test_str_repr)
    unnamed_span = bnl.TimeSpan(start=1.0, end=2.0)
    assert str(unnamed_span) == "TimeSpan([1.00s-2.00s], 1.00s)"
    assert repr(unnamed_span) == "TimeSpan(start=1.0, end=2.0, name='None')"


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

    # Test Segmentation plotting with no segments - No longer possible to create empty_seg
    # empty_seg = bnl.Segmentation()
    # fig, ax = empty_seg.plot()
    # plt.close(fig)

    # Test Hierarchy plotting
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"])
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 2.0], ["a", "b"], name="FineLayer")
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2], name="TestHierPlot")
    fig_single_axis, _ = hierarchy.plot_single_axis() # plot_single_axis still returns fig, ax
    plt.close(fig_single_axis)
    fig_subplots = hierarchy.plot() # hierarchy.plot() now returns only fig
    plt.close(fig_subplots)


    # Test Hierarchy plotting with one layer
    hierarchy_single_layer = bnl.Hierarchy(layers=[seg1], name="SingleLayerHier")
    fig_single_axis_single, _ = hierarchy_single_layer.plot_single_axis()
    plt.close(fig_single_axis_single)
    fig_subplots_single = hierarchy_single_layer.plot()
    plt.close(fig_subplots_single)

    # Test plotting with style_map
    fig, ax = span_named.plot(color="red", ymax=0.5)
    plt.close(fig)

    # Test plotting hierarchy with empty layers - Instantiation will fail first
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        empty_hierarchy_sa = bnl.Hierarchy()
        # empty_hierarchy_sa.plot_single_axis() # This line won't be reached
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        empty_hierarchy_pl = bnl.Hierarchy()
        # empty_hierarchy_pl.plot() # This line won't be reached


# Fixture to load JAMS file content for tests
@pytest.fixture
def test_seg_jams_content():
    path = Path(__file__).parent / "fixtures" / "annotations" / "test_seg.jams"
    with open(path) as f:
        return f.read()

@pytest.fixture
def test_hier_jams_content():
    path = Path(__file__).parent / "fixtures" / "annotations" / "test_hier.jams"
    with open(path) as f:
        return f.read()

@pytest.fixture
def test_hier_json_content():
    path = Path(__file__).parent / "fixtures" / "annotations" / "test_hier.json"
    with open(path) as f:
        return json.load(f) # JSON fixture should be loaded as dict/list


def test_segmentation_from_jams(test_seg_jams_content):
    """Test creating a Segmentation from a JAMS annotation using a real JAMS file."""
    jam = jams.JAMS.loads(test_seg_jams_content)
    anno = jam.annotations.search(namespace="segment_open")[0]

    seg = bnl.Segmentation.from_jams(anno)
    assert len(seg) == 2
    assert seg.name == "segment_open" # Namespace becomes the name
    assert seg.segments[0].start == 0.0
    assert seg.segments[0].end == 5.0
    assert seg.segments[0].name == "verse"
    assert seg.segments[1].start == 5.0
    assert seg.segments[1].end == 10.0
    assert seg.segments[1].name == "chorus"
    assert seg.duration == 10.0

    # Test with an annotation that is not for segmentation (e.g. event-like)
    # For this, we can create a simple beat annotation in the JAMS object
    beat_data = [
        jams.Observation(time=0.0, duration=0.0, value=1, confidence=None),
        jams.Observation(time=1.0, duration=0.0, value=2, confidence=None),
        jams.Observation(time=2.0, duration=0.0, value=3, confidence=None),
    ]
    beat_anno = jams.Annotation(namespace='beat', data=beat_data)
    seg_beat = bnl.Segmentation.from_jams(beat_anno)
    assert seg_beat.name == 'beat'
    assert len(seg_beat) == 3
    assert seg_beat.segments[0].name == '1' # Value is cast to str in TimeSpan
    assert np.isclose(seg_beat.segments[1].start, 1.0)
    # Contiguity check should be relaxed for 'beat' namespace
    assert np.isclose(seg_beat.start, 0.0)
    assert np.isclose(seg_beat.end, 2.0) # End is the end of the last segment (start + duration)


def test_hierarchy_from_jams(test_hier_jams_content):
    """Test creating a Hierarchy from a JAMS multi_segment annotation using a real JAMS file."""
    jam = jams.JAMS.loads(test_hier_jams_content)
    multi_segment_anno = jam.annotations.search(namespace="multi_segment")[0]

    hierarchy = bnl.Hierarchy.from_jams(multi_segment_anno, name="MyTestHier")
    assert hierarchy.name == "MyTestHier" # Explicit name
    assert len(hierarchy) == 2
    assert hierarchy.duration == 10.0

    # Check layer 0 (coarsest)
    assert hierarchy.layers[0].name == "level_0" # Default layer name
    assert len(hierarchy.layers[0]) == 1
    assert hierarchy.layers[0].segments[0].name == "A"
    assert hierarchy.layers[0].segments[0].start == 0.0
    assert hierarchy.layers[0].segments[0].end == 10.0

    # Check layer 1 (finer)
    assert hierarchy.layers[1].name == "level_1" # Default layer name
    assert len(hierarchy.layers[1]) == 2
    assert hierarchy.layers[1].segments[0].name == "a"
    assert hierarchy.layers[1].segments[0].start == 0.0
    assert hierarchy.layers[1].segments[0].end == 5.0
    assert hierarchy.layers[1].segments[1].name == "b"
    assert hierarchy.layers[1].segments[1].start == 5.0
    assert hierarchy.layers[1].segments[1].end == 10.0

    # Test default hierarchy name from JAMS metadata if no name is passed
    hierarchy_default_name = bnl.Hierarchy.from_jams(multi_segment_anno)
    assert hierarchy_default_name.name == "Test Annotator" # From annotator.name

    # Test wrong namespace
    jam.annotations[0].namespace = "wrong_namespace" # Modify fixture for this test case
    with pytest.raises(ValueError, match="Expected 'multi_segment' namespace"):
        bnl.Hierarchy.from_jams(jam.annotations[0])


# def test_hierarchy_not_implemented_constructors():
#     """Test that from_boundaries, from_intervals raise NotImplementedError."""
#     # bnl.Hierarchy.from_json is now implemented.
#     # These are currently commented out in core.py
#     with pytest.raises(NotImplementedError):
#         bnl.Hierarchy.from_boundaries([[]])
#     with pytest.raises(NotImplementedError):
#         bnl.Hierarchy.from_intervals([np.array([])])


def test_hierarchy_from_json_adobe_est_format(test_hier_json_content): # Added fixture argument
    """Test creating a Hierarchy from the Adobe EST JSON structure, using the fixture."""
    json_data_valid = test_hier_json_content
    # Expected structure from test_hier.json:
    # [
    #     [[[0.0, 10.0]], ["A"]],
    #     [[[0.0, 5.0], [5.0, 10.0]], ["a", "b"]]
    # ]
    hierarchy = bnl.Hierarchy.from_json(json_data_valid, name="TestAdobeESTHierarchy")

    assert hierarchy.name == "TestAdobeESTHierarchy"
    assert len(hierarchy) == 2
    assert hierarchy.duration == 10.0

    # Layer 0
    assert hierarchy.layers[0].name == "layer_0" # Default layer name
    assert len(hierarchy.layers[0]) == 1
    assert hierarchy.layers[0].segments[0].name == "A"
    assert hierarchy.layers[0].segments[0].start == 0.0
    assert hierarchy.layers[0].segments[0].end == 10.0

    # Layer 1
    assert hierarchy.layers[1].name == "layer_1" # Default layer name
    assert len(hierarchy.layers[1]) == 2
    assert hierarchy.layers[1].segments[0].name == "a"
    assert hierarchy.layers[1].segments[0].start == 0.0
    assert hierarchy.layers[1].segments[0].end == 5.0
    assert hierarchy.layers[1].segments[1].name == "b"
    assert hierarchy.layers[1].segments[1].start == 5.0
    assert hierarchy.layers[1].segments[1].end == 10.0

    # Test more complex data directly (different from fixture)
    json_data_complex_layers = [
        [[[0.0, 12.0]], ["FullSpan"]], # Layer 0
        [[[0.0, 4.0], [4.0, 8.0], [8.0, 12.0]], ["P1", "P2", "P3"]], # Layer 1
        [ # Layer 2: more segments
            [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0], [10.0, 12.0]],
            ["s1", "s2", "s3", "s4", "s5", "s6"],
        ],
    ]
    h_complex = bnl.Hierarchy.from_json(json_data_complex_layers, name="ComplexJSONHier")
    assert h_complex.name == "ComplexJSONHier"
    assert len(h_complex) == 3
    assert h_complex.duration == 12.0
    assert h_complex.layers[0].name == "layer_0"
    assert len(h_complex.layers[0].segments) == 1
    assert h_complex.layers[1].name == "layer_1"
    assert len(h_complex.layers[1].segments) == 3
    assert h_complex.layers[2].name == "layer_2"
    assert len(h_complex.layers[2].segments) == 6
    assert h_complex.layers[2].segments[5].name == "s6"
    assert h_complex.layers[2].segments[5].end == 12.0

    # Test with the more complex interval structure [[start, end]] per segment
    json_data_complex_interval = [
        [[[0.0, 10.0]], ["Layer0_SegA"]],
    ]
    hierarchy_complex_interval = bnl.Hierarchy.from_json(json_data_complex_interval)
    assert len(hierarchy_complex_interval.layers[0]) == 1
    assert hierarchy_complex_interval.layers[0].segments[0].start == 0.0
    assert hierarchy_complex_interval.layers[0].segments[0].end == 10.0

    # Test with empty list (no layers) - Now raises ValueError
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        bnl.Hierarchy.from_json([])

    # Test with a layer that has no segments (empty intervals_list and labels_list)
    # This will fail when trying to create an empty Segmentation for that layer.
    json_data_empty_layer = [
        [[[0.0, 10.0]], ["Layer0_SegA"]], # Valid layer
        [[], []],  # This layer will cause Segmentation init to fail
        [[[0.0, 10.0]], ["Layer2_SegA"]], # Valid layer (won't be reached)
    ]
    with pytest.raises(ValueError, match="Segmentation must contain at least one segment."):
        bnl.Hierarchy.from_json(json_data_empty_layer)

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
            [[[0.0, 12.0]], ["b"]], # Layer 0 end 10.0, Layer 1 end 12.0
    ]
    # Updated match based on the more specific error message from Hierarchy.__post_init__
    with pytest.raises(ValueError, match=r"All non-empty layers in a Hierarchy must span .* time range\."):
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

    expected_yticks = ["L1", "L0"] # Updated to expect layer names
    actual_yticks = [label.get_text() for label in ax.get_yticklabels()]
    assert actual_yticks == expected_yticks

    plt.close(fig)

    # Test with a layer that has no name (should default to "Level X")
    seg_no_name = bnl.Segmentation.from_boundaries([0.0, 1.0, 4.0], ["n1", "n2"]) # No name for seg_no_name
    hierarchy_with_mixed_names = bnl.Hierarchy(layers=[seg1, seg_no_name, seg2], name="MixedNames") # seg1 is L0, seg2 is L1

    fig_mixed, ax_mixed = hierarchy_with_mixed_names.plot_single_axis()
    # Order of layers: L0, seg_no_name, L1. Reversed for yticks: L1, Level 1 (for seg_no_name), L0
    expected_yticks_mixed = ["L1", "Level 1", "L0"]
    actual_yticks_mixed = [label.get_text() for label in ax_mixed.get_yticklabels()]
    assert actual_yticks_mixed == expected_yticks_mixed
    plt.close(fig_mixed)

    # The following sub-test is invalid as seg_empty cannot be created without segments.
    # seg_empty = bnl.Segmentation(start=0.0, end=4.0, name="EmptyLayer")
    # hierarchy_with_empty = bnl.Hierarchy(layers=[seg1, seg_empty, seg2])
    # fig_empty, ax_empty = hierarchy_with_empty.plot_single_axis()
    # assert len(ax_empty.get_yticklabels()) == 3
    # expected_yticks_empty = ["L1", "EmptyLayer", "L0"]
    # actual_yticks_empty = [label.get_text() for label in ax_empty.get_yticklabels()]
    # assert actual_yticks_empty == expected_yticks_empty
    # plt.close(fig_empty)

    # Instantiation of empty Hierarchy will fail before plotting
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        empty_hierarchy_plot_single = bnl.Hierarchy()
        # empty_hierarchy_plot_single.plot_single_axis() # Not reached
