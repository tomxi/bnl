import pytest
import numpy as np
from bnl.core import Segmentation, multi_to_segmentation
from bnl.formats import mireval2multi


@pytest.fixture(scope="module")
def hierarchies():
    ITVLS1 = np.array([[0, 2.5], [2.5, 6.01]])
    LABELS1 = ["A", "B"]

    ITVLS2 = np.array([[0, 1.2], [1.2, 2.5], [2.5, 3.5], [3.5, 6.01]])
    LABELS2 = ["a", "b", "c", "b"]

    ITVLS3 = np.array([[0, 1.2], [1.2, 4], [4, 6.01]])
    LABELS3 = ["Mi", "Re", "Do"]

    ITVLS4 = np.array([[0, 1.2], [1.2, 3], [3, 4], [4, 6.01]])
    LABELS4 = ["T", "PD", "D", "T"]

    ITVLS5 = np.array(
        [[0, 1.2], [1.2, 2], [2, 3], [3, 4], [4, 4.7], [4.7, 5.3], [5.3, 6.01]]
    )
    LABELS5 = ["I", "IV", "ii", "V", "I", "IV", "I"]

    # Hierarchical segmentations
    hier1 = Segmentation([ITVLS1, ITVLS2], [LABELS1, LABELS2], is_hierarchical=True)
    hier2 = Segmentation([ITVLS3, ITVLS4, ITVLS5], [LABELS3, LABELS4, LABELS5], is_hierarchical=True)
    hier3 = Segmentation(
        [ITVLS1, ITVLS2, ITVLS3, ITVLS4, ITVLS5],
        [LABELS1, LABELS2, LABELS3, LABELS4, LABELS5],
        is_hierarchical=True
    )

    # Flat segmentations (taking the first level of hierarchical for simplicity in tests)
    flat1 = Segmentation(ITVLS1, LABELS1, is_hierarchical=False)
    flat2 = Segmentation(ITVLS2, LABELS2, is_hierarchical=False)

    return dict(h1=hier1, h2=hier2, h3=hier3, f1=flat1, f2=flat2)


def test_flat_segmentation_initialization(hierarchies):
    seg = hierarchies["f1"] # Use a pre-made flat segmentation

    assert np.allclose(seg.itvls, np.array([[0, 2.5], [2.5, 6.01]])) # ITVLS1
    assert seg.Lstar[0] == "A" # First label of LABELS1
    assert seg.T0 == 0
    assert seg.T == 6.01 # End time of ITVLS1


@pytest.mark.parametrize("text", [True, False])
def test_flat_segmentation_ploting_parametrized(text, hierarchies):
    flat_seg = hierarchies["f1"]
    fig_tuple = flat_seg.plot(text=text) # plot returns fig, ax or fig, axs
    assert fig_tuple is not None
    assert fig_tuple[0] is not None # Check figure object itself


def test_segmentation_call_flat(hierarchies):
    seg = hierarchies["f2"] # LABELS2: ["a", "b", "c", "b"]
    assert seg(1) == "a"
    assert seg(3) == "c"
    assert seg(3.5) == "b" # Boundary, should get label of segment starting at 3.5
    assert seg(5.0) == "b"


# expected fail with Index Error decoration
@pytest.mark.xfail(raises=IndexError)
def test_segmentation_call_flat_out_of_bounds(hierarchies):
    seg = hierarchies["f1"]
    seg(10) # Call __call__


def test_segmentation_B_flat(hierarchies):
    seg = hierarchies["f1"] # ITVLS1: [[0, 2.5], [2.5, 6.01]]
    assert seg.B(0) == 1
    assert seg.B(2.5) == 1
    assert seg.B(1.5) == 0
    assert seg.B(seg.T) == 1 # Boundary at T
    assert seg.B(seg.T + 1) == 0 # Outside


def test_segmentation_Bhat_flat(hierarchies):
    seg = hierarchies["f1"]
    seg.update_sr(20) # Ensure sr and ticks are set for Bhat
    bhat_values = seg.Bhat(ts=np.array([1, 2, 3]))
    assert len(bhat_values) == 3


def test_segmentation_Ahat_flat(hierarchies):
    seg = hierarchies["f1"]
    seg.update_sr(20) # Ensure sr and ticks are set
    ahat_matrix = seg.Ahat()
    # Dimensions should be (num_segments-1, num_segments-1) if using default beta, or based on bs
    # For f1 (2 segments), beta has 3 points [0, 2.5, 6.01]. Ahat based on midpoints of these.
    # So, 2 midpoints, Ahat matrix is 2x2.
    # If Ahat is defined on segments, it would be 1x1 for 2 segments.
    # The current S.Ahat logic: labels_arr = np.array([self(t) for t in ts]) where ts are midpoints
    # ts = (bs[1:] + bs[:-1]) / 2. If bs = self.beta = [0, 2.5, 6.01], ts = [1.25, 4.255]
    # labels_arr = [seg(1.25), seg(4.255)] = ["A", "B"]. Ahat will be 2x2.
    assert ahat_matrix.shape == (len(seg.beta)-1, len(seg.beta)-1)

# --- Tests for Hierarchical aspects of Segmentation ---

@pytest.mark.parametrize("text", [True, False])
def test_hierarchical_segmentation_ploting_parametrized(text, hierarchies):
    hier_seg = hierarchies["h1"]
    fig_tuple = hier_seg.plot(text=text)
    assert fig_tuple is not None
    assert fig_tuple[0] is not None # Check figure object
    assert len(fig_tuple[1]) == hier_seg.d # Check number of axes matches depth

def test_segmentation_call_hierarchical(hierarchies):
    h_seg = hierarchies["h1"] # Levels are S(ITVLS1, LABELS1) and S(ITVLS2, LABELS2)
    # For x=1.0: Level 0 ("S1"): in [0, 2.5] -> "A"
    #            Level 1 ("S2"): in [0, 1.2] -> "a"
    assert h_seg(1.0) == ["A", "a"]
    # For x=3.0: Level 0 ("S1"): in [2.5, 6.01] -> "B"
    #            Level 1 ("S2"): in [2.5, 3.5] -> "c"
    assert h_seg(3.0) == ["B", "c"]

@pytest.mark.xfail(raises=IndexError)
def test_segmentation_call_hierarchical_out_of_bounds(hierarchies):
    h_seg = hierarchies["h1"]
    h_seg(10)

def test_segmentation_B_hierarchical(hierarchies):
    h_seg = hierarchies["h1"] # L0: [0, 2.5, 6.01], L1: [0, 1.2, 2.5, 3.5, 6.01]
    # B for hierarchical returns coarsest level_idx+1 where it's a boundary
    assert h_seg.B(0) == h_seg.d # Boundary in all levels (deepest for H)
    assert h_seg.B(2.5) == h_seg.d
    assert h_seg.B(1.2) == h_seg.d - 1 # Only in level 1 (S2), so depth is d-(idx_of_S2) = 2-1 = 1 if S2 is index 1.
                                      # B returns (d-k) where k is level index. If k=1 (level S2), returns d-1.
                                      # If S2 is level 1 (0-indexed), then d-1 = 2-1 = 1.
                                      # If S1 is level 0, S2 is level 1. Coarsest level S1 (idx 0).
                                      # B(t) returns d-k. k is 0-indexed.
                                      # B(1.2) is in S2 (level[1]), so k=1. Returns d-1 = 2-1=1.
                                      # This means it's a boundary up to level 1 (from the top, 1-indexed).
                                      # The original H.B logic was: "coarsest level (d-k) that a boundary appears in"
                                      # So if it appears in S2 (k=1), salience is d-1. If in S1 (k=0), salience is d.
    assert h_seg.B(3.5) == h_seg.d -1 # Only in S2 (level[1])
    assert h_seg.B(1.5) == 0 # Not a boundary in any level
    assert h_seg.B(h_seg.T) == h_seg.d

def test_segmentation_Astar(hierarchies):
    h_seg = hierarchies["h1"]
    h_seg.update_sr(20) # Ensure sr and ticks for all levels
    astar_matrix = h_seg.Astar()
    # Astar is based on self.beta which is union of all level betas.
    # beta for h1: unique([0,2.5,6.01] + [0,1.2,2.5,3.5,6.01]) = [0,1.2,2.5,3.5,6.01] (5 points)
    # So Astar matrix should be (5-1)x(5-1) = 4x4
    assert astar_matrix.shape == (len(h_seg.beta)-1, len(h_seg.beta)-1)

def test_hierarchical_specific_method_on_flat_fails(hierarchies):
    flat_seg = hierarchies["f1"]
    with pytest.raises(TypeError):
        flat_seg.Astar()
    with pytest.raises(TypeError):
        flat_seg.Ahats()
    with pytest.raises(TypeError):
        flat_seg.Bhats()
    with pytest.raises(TypeError):
        flat_seg.M()
    with pytest.raises(TypeError):
        flat_seg.Mhat()
    with pytest.raises(TypeError):
        flat_seg.decode_B()
    with pytest.raises(TypeError):
        flat_seg.decode_L([])
    with pytest.raises(TypeError):
        flat_seg.decode()

from bnl.core import flat_to_segmentation # Import the function

# --- Tests for Segmentation Class helper functions ---
def test_multi_to_segmentation(hierarchies):
    # hierarchies['h1'] is a Segmentation object. Its .anno attribute is a JAMS annotation.
    jams_anno = hierarchies['h1'].anno
    seg_from_multi = multi_to_segmentation(jams_anno, sr=20, Bhat_bw=0.5)
    assert seg_from_multi.is_hierarchical
    assert seg_from_multi.d == hierarchies['h1'].d
    assert len(seg_from_multi.levels) == hierarchies['h1'].d
    assert seg_from_multi.sr == 20
    assert seg_from_multi.Bhat_bw == 0.5
    # Compare structure - e.g. number of segments in each level
    for l_idx in range(seg_from_multi.d):
        assert len(seg_from_multi.levels[l_idx].itvls) == len(hierarchies['h1'].levels[l_idx].itvls)

def test_flat_to_segmentation(hierarchies):
    # hierarchies['f1'] is a flat Segmentation. Its .anno is a JAMS annotation.
    jams_anno_flat = hierarchies['f1'].anno
    seg_from_flat = flat_to_segmentation(jams_anno_flat, sr=22, Bhat_bw=0.7)
    assert not seg_from_flat.is_hierarchical
    assert seg_from_flat.d == 1
    assert seg_from_flat.sr == 22
    assert seg_from_flat.Bhat_bw == 0.7
    assert len(seg_from_flat.itvls) == len(hierarchies['f1'].itvls)

# More tests can be added for levels_to_segmentation and sal_to_segmentation
# For sal_to_segmentation, it might require a prune_identical_levels method in Segmentation
# or careful construction of test salience data.

def test_segmentation_expand_flat(hierarchies):
    flat_seg = hierarchies["f1"] # Has 2 segments
    # expand should create a hierarchical segmentation from a flat one
    # by default, expand_hierarchy in .hierarchy (called by Segmentation.expand)
    # might create 2 or 3 levels if labels can be flattened/expanded.
    # For "A", "B", flatten_labels might keep them as is. expand_labels will make "A_0", "B_0".
    # So, original, and expanded. Potentially flattened if different.
    expanded_seg = flat_seg.expand(format="slm", always_include=True) # Use always_include=True
    assert expanded_seg.is_hierarchical
    # Check that the original labels are present in one of the levels
    original_labels_present = any(
        np.array_equal(level.labels, flat_seg.labels) for level in expanded_seg.levels
    )
    assert original_labels_present
    # Check if expanded labels like "A_0", "B_0" are present
    expanded_labels_present = any(
        level.labels == ["A_0", "B_0"] for level in expanded_seg.levels
    )
    assert expanded_labels_present


def test_segmentation_expand_hierarchical(hierarchies):
    hier_seg = hierarchies["h1"] # Has 2 levels
    # Expanding a hierarchical segmentation should expand each of its levels
    # and combine them into a new, potentially deeper, hierarchy.
    expanded_hier_seg = hier_seg.expand(format="slm")
    assert expanded_hier_seg.is_hierarchical
    # The number of levels should be >= original number of levels
    assert expanded_hier_seg.d >= hier_seg.d
    # Check if original levels' structures (intervals) are preserved among the expanded levels
    # This is tricky because expand_hierarchy creates new JAMS annotations.
    # A simpler check: total number of segments in all expanded levels vs original.
    total_original_segments = sum(len(level.itvls) for level in hier_seg.levels)
    total_expanded_segments = sum(len(level.itvls) for level in expanded_hier_seg.levels)
    # Each original level contributes at least its own segments to the expanded version.
    assert total_expanded_segments >= total_original_segments
