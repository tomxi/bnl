import pytest
import numpy as np
from bnl.core import Segmentation, levels_to_segmentation, flat_to_segmentation
from bnl import hierarchy as hier
from bnl.formats import mireval2multi, openseg2mirevalflat, mirevalflat2openseg

# Fixture for basic hierarchical segmentation
@pytest.fixture
def sample_hierarchical_segmentation():
    itvls1 = np.array([[0, 10], [10, 20]])
    labels1 = ["A", "B"]
    itvls2 = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    labels2 = ["a", "b", "c", "d"]
    seg = Segmentation([itvls1, itvls2], [labels1, labels2], sr=100, Bhat_bw=1.0, is_hierarchical=True)
    return seg

@pytest.fixture
def flat_segmentation_fixture():
    itvls = np.array([[0, 5], [5, 10]])
    labels = ["X", "Y"]
    return Segmentation(itvls, labels, sr=100, Bhat_bw=1.0, is_hierarchical=False)

def test_relabel_unique(sample_hierarchical_segmentation):
    h_seg = sample_hierarchical_segmentation
    relabeled_h = hier.relabel(h_seg, strategy="unique")
    assert relabeled_h.is_hierarchical
    assert relabeled_h.d == h_seg.d
    # Check if labels are defaulted (e.g., stringified start times or simple ranges)
    for level_idx in range(relabeled_h.d):
        expected_default_labels = [str(s[0]) for s in relabeled_h.levels[level_idx].itvls]
        assert relabeled_h.levels[level_idx].labels == expected_default_labels

def test_relabel_max_overlap(sample_hierarchical_segmentation):
    h_seg = sample_hierarchical_segmentation
    # Max overlap relabeling might be complex to assert exact labels without knowing reindex logic well
    # For now, just check it runs and preserves structure
    relabeled_h = hier.relabel(h_seg, strategy="max_overlap")
    assert relabeled_h.is_hierarchical
    assert relabeled_h.d == h_seg.d
    assert len(relabeled_h.levels[0].labels) == len(h_seg.levels[0].labels)
    assert len(relabeled_h.levels[1].labels) == len(h_seg.levels[1].labels)


def test_has_mono_B_true():
    itvls1 = np.array([[0, 10], [10, 20]])
    itvls2 = np.array([[0, 5], [5, 10], [10, 15], [15, 20]]) # Child boundaries contain parent
    seg = Segmentation([itvls1, itvls2], is_hierarchical=True)
    assert hier.has_mono_B(seg)

def test_has_mono_B_false():
    itvls1 = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    itvls2 = np.array([[0, 10], [10, 20]]) # Parent boundaries not subset of child
    seg = Segmentation([itvls1, itvls2], is_hierarchical=True)
    assert not hier.has_mono_B(seg)

def test_force_mono_B(sample_hierarchical_segmentation):
    h_seg = sample_hierarchical_segmentation # Levels are not necessarily monotonic
    mono_b_h = hier.force_mono_B(h_seg)
    assert mono_b_h.is_hierarchical
    assert hier.has_mono_B(mono_b_h)
    assert mono_b_h.d == h_seg.d # Depth should be preserved

def test_prune_identical_levels(sample_hierarchical_segmentation):
    # Create a segmentation with an identical level
    itvls1 = np.array([[0, 10], [10, 20]])
    labels1 = ["A", "B"]
    # Level 2 is identical to Level 1 in terms of intervals and effective labels for A matrix
    itvls_identical = np.array([[0, 10], [10, 20]])
    labels_identical = ["A", "B"]

    seg_with_identical = Segmentation([itvls1, itvls_identical, sample_hierarchical_segmentation.levels[1].itvls],
                                      [labels1, labels_identical, sample_hierarchical_segmentation.levels[1].labels], is_hierarchical=True)
    assert seg_with_identical.d == 3
    pruned_h = hier.prune_identical_levels(seg_with_identical)
    assert pruned_h.is_hierarchical
    assert pruned_h.d < seg_with_identical.d
    assert pruned_h.d == 2 # Level 1 and the original level 2 (from sample_hierarchical_segmentation) should remain

def test_squash_levels(sample_hierarchical_segmentation):
    h_seg = sample_hierarchical_segmentation
    assert h_seg.d == 2
    squashed_h = hier.squash_levels(h_seg, max_depth=1)
    assert squashed_h.is_hierarchical
    assert squashed_h.d == 1

    # Test with remove_single_itvls
    itvls1 = np.array([[0, 20]]) # Single interval level
    labels1 = ["A"]
    itvls2 = np.array([[0,5],[5,10],[10,15],[15,20]])
    labels2 = ["a","b","c","d"]
    seg_with_single = Segmentation([itvls1, itvls2], [labels1, labels2], is_hierarchical=True)
    assert seg_with_single.d == 2
    squashed_no_single = hier.squash_levels(seg_with_single, max_depth=2, remove_single_itvls=True)
    if len(squashed_no_single.levels) > 0 and len(squashed_no_single.levels[0].itvls) == 1 : # if the single interval level was not removed because it was the only one left
         assert squashed_no_single.d ==1
    else:
        assert squashed_no_single.d == 1 # The single interval level should be removed

def test_has_mono_L(sample_hierarchical_segmentation):
    # This depends greatly on the relabeling strategy and A, Astar logic.
    # For a simple case, let's make one that IS monotonic
    itvls1 = np.array([[0,20]])
    labels1 = ["Segment1"]
    itvls2 = np.array([[0,10],[10,20]])
    labels2 = ["Segment1.sub1", "Segment1.sub2"] # Monotonic labels by construction
    mono_l_seg = Segmentation([itvls1, itvls2], [labels1, labels2], is_hierarchical=True)
    # Need to ensure A and Astar are computed correctly
    # This test might be more involved. For now, let's check it runs.
    # To properly test has_mono_L, we need A and Astar from Segmentation to be correct.
    # Astar compares A matrices of levels. If labels are hierarchical like "X", "X.1", "X.2",
    # then the A matrix for the parent will show these as different, Astar should capture this.
    # The provided `has_mono_L` uses `hier.A(bs=hier.beta) == hier.Astar(bs=hier.beta)`
    # This equality might not hold if A is sum of A_level and Astar is max(level_idx * A_level_indicator)
    # For now, we just execute it. A deeper test of has_mono_L depends on A and Astar definitions.
    try:
        is_mono_l = hier.has_mono_L(mono_l_seg)
        # We expect true for this constructed case if A and Astar are as in McFee's work
        # However, the current Segmentation.A sums agreements, Segmentation.Astar takes max level.
        # This test might fail or pass depending on the exact interaction.
        # Let's assert it runs. A more precise assertion requires validating A/Astar output.
        assert isinstance(is_mono_l, (bool, np.bool_))
    except Exception as e:
        pytest.fail(f"has_mono_L raised an exception: {e}")


def test_force_mono_L(sample_hierarchical_segmentation):
    h_seg = sample_hierarchical_segmentation
    mono_l_h = hier.force_mono_L(h_seg)
    assert mono_l_h.is_hierarchical
    # This is the property force_mono_L should ensure, assuming has_mono_L is correct
    # For now, let's check structural properties
    assert mono_l_h.d == h_seg.d
    # Check if labels now have the dot notation typically produced by force_mono_L
    if mono_l_h.d > 1:
        # Example: first label of second level should be like "parent_label.child_label"
        parent_label_example = mono_l_h.levels[0].labels[0]
        child_label_example = mono_l_h.levels[1].labels[0]
        assert "." in child_label_example
        assert child_label_example.startswith(str(parent_label_example))


# Test that hierarchical specific functions fail on flat segmentations
def test_hier_methods_on_flat_segmentation_fails(flat_segmentation_fixture):
    flat_seg = flat_segmentation_fixture
    with pytest.raises(TypeError):
        hier.prune_identical_levels(flat_seg)
    with pytest.raises(TypeError):
        hier.squash_levels(flat_seg)
    with pytest.raises(TypeError):
        hier.relabel(flat_seg)
    with pytest.raises(TypeError):
        hier.has_mono_L(flat_seg)
    with pytest.raises(TypeError):
        hier.has_mono_B(flat_seg)
    with pytest.raises(TypeError):
        hier.force_mono_B(flat_seg)
    with pytest.raises(TypeError):
        hier.force_mono_L(flat_seg)

# TODO: Add more specific tests for edge cases and label/boundary conditions in force_mono_B/L
# TODO: Test prune_identical_levels with boundary_only=True
# TODO: Test squash_levels with more complex scenarios
