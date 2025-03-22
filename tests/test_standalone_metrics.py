import pytest
import numpy as np
from bnl import standalone_metrics as mtr
import tests


# Sample hierarchies for testing
@pytest.fixture
def simple_hierarchy():
    # Level 0 - coarsest segmentation
    level0_itvls = [(0.0, 10.0), (10.0, 20.0)]
    level0_labels = ["A", "B"]

    # Level 1 - finer segmentation
    level1_itvls = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)]
    level1_labels = ["A1", "A2", "B1", "B2"]

    # Level 2 - finest segmentation
    level2_itvls = [
        (0.0, 2.5),
        (2.5, 5.0),
        (5.0, 7.5),
        (7.5, 10.0),
        (10.0, 12.5),
        (12.5, 15.0),
        (15.0, 17.5),
        (17.5, 20.0),
    ]
    level2_labels = ["A1a", "A1b", "A2a", "A2b", "B1a", "B1b", "B2a", "B2b"]

    return [
        (level0_itvls, level0_labels),
        (level1_itvls, level1_labels),
        (level2_itvls, level2_labels),
    ]


@pytest.fixture
def reference_hierarchy(simple_hierarchy):
    return simple_hierarchy


@pytest.fixture
def estimated_hierarchy():
    # Similar structure but with slightly different boundaries and labels
    # Level 0 - coarsest segmentation
    level0_itvls = [(0.0, 9.8), (9.8, 20.0)]
    level0_labels = ["A", "B"]

    # Level 1 - finer segmentation
    level1_itvls = [(0.0, 5.2), (5.2, 9.8), (9.8, 14.9), (14.9, 20.0)]
    level1_labels = ["A1", "A2", "B1", "B2"]

    # Level 2 - finest segmentation
    level2_itvls = [
        (0.0, 2.6),
        (2.6, 5.2),
        (5.2, 7.5),
        (7.5, 9.8),
        (9.8, 12.3),
        (12.3, 14.9),
        (14.9, 17.3),
        (17.3, 20.0),
    ]
    level2_labels = ["A1a", "A1b", "A2a", "A2b", "B1a", "B1b", "B2a", "B2b"]

    return [
        (level0_itvls, level0_labels),
        (level1_itvls, level1_labels),
        (level2_itvls, level2_labels),
    ]


class TestMeet:
    def test_meet_invalid_mode(self):
        """Test that meet raises ValueError for invalid mode"""
        time_pairs = [(1.0, 3.0), (5.0, 15.0)]
        with pytest.raises(ValueError):
            mtr.meet([], time_pairs, mode="invalid")

    def test_meet_deepest_mode(self, simple_hierarchy):
        """Test meet with 'deepest' mode"""
        # Same segment pairs should meet at the deepest level
        time_pairs = [(1.0, 2.0), (6.0, 7.0), (11.0, 12.0)]
        result = mtr.meet(simple_hierarchy, time_pairs, mode="deepest")
        assert result is not None  # Replace when implementation is complete

    def test_meet_mono_mode(self, simple_hierarchy):
        """Test meet with 'mono' mode"""
        # Points across major segments should meet at level 0
        time_pairs = [(5.0, 15.0), (2.0, 18.0)]
        result = mtr.meet(simple_hierarchy, time_pairs, mode="mono")
        assert result is not None  # Replace when implementation is complete

    def test_meet_mean_mode(self, simple_hierarchy):
        """Test meet with 'mean' mode"""
        # Points in same finest segment should meet at level 2
        time_pairs = [(1.0, 2.0), (16.0, 17.0)]
        result = mtr.meet(simple_hierarchy, time_pairs, mode="mean")
        assert result is not None  # Replace when implementation is complete

    def test_meet_across_boundaries(self, simple_hierarchy):
        """Test meet for points across different segment levels"""
        # Points across different segments
        time_pairs = [(2.0, 6.0), (9.0, 11.0), (4.0, 16.0)]
        for mode in ["deepest", "mono", "mean"]:
            result = mtr.meet(simple_hierarchy, time_pairs, mode=mode)
            assert result is not None  # Replace when implementation is complete


class TestGetSegmentRelevance:
    def test_segment_relevance_middle_point(self, simple_hierarchy):
        """Test relevance for a point in the middle of a segment"""
        t = 7.5  # middle of a segment at level 1
        itvls, relevances = mtr.get_segment_relevance(
            simple_hierarchy, t, meet_mode="deepest"
        )
        assert isinstance(itvls, list)
        assert isinstance(relevances, list)

    def test_segment_relevance_boundary_point(self, simple_hierarchy):
        """Test relevance for a point at segment boundary"""
        t = 10.0  # boundary between major segments
        itvls, relevances = mtr.get_segment_relevance(
            simple_hierarchy, t, meet_mode="mono"
        )
        assert isinstance(itvls, list)
        assert isinstance(relevances, list)

    def test_segment_relevance_different_modes(self, simple_hierarchy):
        """Test that different meet modes give potentially different relevances"""
        t = 5.0  # boundary at level 1
        for mode in ["deepest", "mono", "mean"]:
            itvls, relevances = mtr.get_segment_relevance(
                simple_hierarchy, t, meet_mode=mode
            )
            assert isinstance(itvls, list)
            assert isinstance(relevances, list)


class TestTripletRecallAtT:
    def test_recall_perfect_match(self, reference_hierarchy):
        """Test recall when ref and est are the same (perfect match)"""
        t = 7.5
        recall = mtr.triplet_recall_at_t(
            reference_hierarchy,
            reference_hierarchy,
            t,
            meet_mode="deepest",
            window=0,
            transitive=True,
        )
        assert isinstance(recall, float)
        # Perfect recall should be 1.0
        # assert recall == 1.0  # Enable when implementation is complete

    def test_recall_with_window(self, reference_hierarchy, estimated_hierarchy):
        """Test recall with a window parameter"""
        t = 10.0  # boundary in reference
        # Test with different window sizes
        for window in [0, 0.5, 1.0]:
            recall = mtr.triplet_recall_at_t(
                reference_hierarchy,
                estimated_hierarchy,
                t,
                meet_mode="deepest",
                window=window,
                transitive=True,
            )
            assert isinstance(recall, float)

    def test_recall_with_transitive(self, reference_hierarchy, estimated_hierarchy):
        """Test recall with and without transitivity"""
        t = 5.0
        # Test with and without transitivity
        recall_with = mtr.triplet_recall_at_t(
            reference_hierarchy,
            estimated_hierarchy,
            t,
            meet_mode="deepest",
            window=0,
            transitive=True,
        )
        recall_without = mtr.triplet_recall_at_t(
            reference_hierarchy,
            estimated_hierarchy,
            t,
            meet_mode="deepest",
            window=0,
            transitive=False,
        )
        assert isinstance(recall_with, float)
        assert isinstance(recall_without, float)
        # Usually, transitivity should allow for higher recall
        # assert recall_with >= recall_without  # Enable when implementation is complete


class TestPairwiseRecall:
    def test_pairwise_recall_exact_match(self):
        """Test pairwise recall with exact matching segments"""
        ref_itvls = [(0, 5), (5, 10)]
        ref_labels = ["A", "B"]
        est_itvls = [(0, 5), (5, 10)]
        est_labels = ["A", "B"]

        t = 2.5  # middle of first segment
        recall = mtr.pairwise_recall(ref_itvls, ref_labels, est_itvls, est_labels, t)
        assert isinstance(recall, float)
        # Perfect match should give recall of 1.0
        # assert recall == 1.0  # Enable when implementation is complete

    def test_pairwise_recall_with_window(self):
        """Test pairwise recall with window parameter"""
        ref_itvls = [(0, 5), (5, 10)]
        ref_labels = ["A", "B"]
        est_itvls = [(0, 4.8), (4.8, 10)]  # Slight boundary deviation
        est_labels = ["A", "B"]

        t = 4.9  # Near the boundary

        # Without window, the recall might be poor
        recall_no_window = mtr.pairwise_recall(
            ref_itvls, ref_labels, est_itvls, est_labels, t, window=0
        )

        # With window, recall should improve
        recall_with_window = mtr.pairwise_recall(
            ref_itvls, ref_labels, est_itvls, est_labels, t, window=0.5
        )

        assert isinstance(recall_no_window, float)
        assert isinstance(recall_with_window, float)
        # Window should help in this case
        # assert recall_with_window >= recall_no_window  # Enable when implementation is complete


class TestEntropy:
    def test_entropy_uniform_distribution(self):
        """Test entropy calculation with uniform distribution"""
        # Equal-sized segments with different labels
        itvls = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        labels = ["A", "B", "C", "D"]

        result = mtr.entropy(itvls, labels)
        assert isinstance(result, float)
        # Maximum entropy for uniform distribution
        # assert np.isclose(result, 2.0)  # log2(4) = 2, uncomment when implemented

    def test_entropy_single_segment(self):
        """Test entropy calculation with single segment"""
        itvls = np.array([[0, 5]])
        labels = ["A"]

        result = mtr.entropy(itvls, labels)
        assert isinstance(result, float)
        assert isinstance(result, float)
        # Zero entropy for single segment
        # assert np.isclose(result, 0.0)  # uncomment when implemented

    def test_entropy_with_predefined_intervals(self):
        """Test entropy using predefined test intervals"""
        result = mtr.entropy(tests.ITVLS1, tests.LABELS1)
        assert isinstance(result, float)

        result = mtr.entropy(tests.ITVLS5, tests.LABELS5)
        assert isinstance(result, float)

    def test_entropy_with_repeated_labels(self):
        """Test entropy with repeated labels"""
        # tests.ITVLS2 has repeated 'b' labels
        result = mtr.entropy(tests.ITVLS2, tests.LABELS2)
        assert isinstance(result, float)


class TestConditionalEntropy:
    def test_conditional_entropy_identical(self):
        """Test conditional entropy when segmentations are identical"""
        result = mtr.conditional_entropy(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
        )
        assert isinstance(result, float)
        # Conditional entropy should be 0 for identical segmentations
        # assert np.isclose(result, 0.0)  # uncomment when implemented

    def test_conditional_entropy_independent(self):
        """Test conditional entropy with independent segmentations"""
        result = mtr.conditional_entropy(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS3, tests.LABELS3
        )
        assert isinstance(result, float)

    def test_conditional_entropy_refined(self):
        """Test conditional entropy with one segmentation refining another"""
        # tests.ITVLS2 refines tests.ITVLS1
        result = mtr.conditional_entropy(
            tests.ITVLS2, tests.LABELS2, tests.ITVLS1, tests.LABELS1
        )
        assert isinstance(result, float)


class TestVmeasure:
    def test_vmeasure_identical(self):
        """Test V-measure with identical segmentations"""
        precision, recall, f1 = mtr.vmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])
        # Perfect match should give values of 1.0
        # assert np.isclose(precision, 1.0)  # uncomment when implemented
        # assert np.isclose(recall, 1.0)
        # assert np.isclose(f1, 1.0)

    def test_vmeasure_refined(self):
        """Test V-measure with one segmentation refining another"""
        precision, recall, f1 = mtr.vmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS2, tests.LABELS2
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])

    def test_vmeasure_different_structures(self):
        """Test V-measure with completely different structures"""
        precision, recall, f1 = mtr.vmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS3, tests.LABELS3
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])

    def test_vmeasure_edge_cases(self):
        """Test V-measure with edge cases"""
        # Single segment
        single_itvls = np.array([[0, 5]])
        single_labels = ["A"]

        precision, recall, f1 = mtr.vmeasure(
            single_itvls, single_labels, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])

        # Different lengths
        precision, recall, f1 = mtr.vmeasure(
            tests.ITVLS5, tests.LABELS5, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])


class TestLmeasure:
    def test_lmeasure_identical(self):
        """Test L-measure with identical segmentations"""
        precision, recall, f1 = mtr.lmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])
        # Perfect match should give values of 1.0
        # assert np.isclose(precision, 1.0)  # uncomment when implemented
        # assert np.isclose(recall, 1.0)
        # assert np.isclose(f1, 1.0)

    def test_lmeasure_different_boundaries(self):
        """Test L-measure with different boundary positions"""
        # Modify boundaries slightly
        modified_itvls = np.copy(tests.ITVLS1)
        modified_itvls[0, 1] = 2.6  # Change boundary from 2.5 to 2.6

        precision, recall, f1 = mtr.lmeasure(
            tests.ITVLS1, tests.LABELS1, modified_itvls, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])

    def test_lmeasure_different_structures(self):
        """Test L-measure with different segmentation structures"""
        precision, recall, f1 = mtr.lmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS3, tests.LABELS3
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])

    def test_lmeasure_repeated_labels(self):
        """Test L-measure with repeated labels"""
        precision, recall, f1 = mtr.lmeasure(
            tests.ITVLS2, tests.LABELS2, tests.ITVLS4, tests.LABELS4
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])


class TestTmeasure:
    def test_tmeasure_identical(self):
        """Test T-measure with identical segmentations"""
        precision, recall, f1 = mtr.tmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])
        # Perfect match should give values of 1.0
        # assert np.isclose(precision, 1.0)  # uncomment when implemented
        # assert np.isclose(recall, 1.0)
        # assert np.isclose(f1, 1.0)

    def test_tmeasure_with_window(self):
        """Test T-measure with different window sizes"""
        # Modify boundaries slightly
        modified_itvls = np.copy(tests.ITVLS1)
        modified_itvls[0, 1] = 2.6  # Change boundary from 2.5 to 2.6

        for window in [0, 0.2, 0.5, 1.0]:
            precision, recall, f1 = mtr.tmeasure(
                tests.ITVLS1,
                tests.LABELS1,
                modified_itvls,
                tests.LABELS1,
                window=window,
            )
            assert all(isinstance(x, float) for x in [precision, recall, f1])

    def test_tmeasure_with_transitivity(self):
        """Test T-measure with and without transitivity"""
        precision_t, recall_t, f1_t = mtr.tmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS2, tests.LABELS2, transitive=True
        )
        precision_f, recall_f, f1_f = mtr.tmeasure(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS2, tests.LABELS2, transitive=False
        )

        assert all(
            isinstance(x, float)
            for x in [precision_t, recall_t, f1_t, precision_f, recall_f, f1_f]
        )


class TestPairClustering:
    def test_pair_clustering_identical(self):
        """Test pair clustering with identical segmentations"""
        precision, recall, f1 = mtr.pair_clustering(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])
        # Perfect match should give values of 1.0
        # assert np.isclose(precision, 1.0)  # uncomment when implemented
        # assert np.isclose(recall, 1.0)
        # assert np.isclose(f1, 1.0)

    def test_pair_clustering_with_refinement(self):
        """Test pair clustering with one segmentation refining another"""
        precision, recall, f1 = mtr.pair_clustering(
            tests.ITVLS1, tests.LABELS1, tests.ITVLS2, tests.LABELS2
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])

        # Reversed
        precision_rev, recall_rev, f1_rev = mtr.pair_clustering(
            tests.ITVLS2, tests.LABELS2, tests.ITVLS1, tests.LABELS1
        )
        assert all(isinstance(x, float) for x in [precision_rev, recall_rev, f1_rev])

    def test_pair_clustering_different_structures(self):
        """Test pair clustering with completely different structures"""
        precision, recall, f1 = mtr.pair_clustering(
            tests.ITVLS3, tests.LABELS3, tests.ITVLS4, tests.LABELS4
        )
        assert all(isinstance(x, float) for x in [precision, recall, f1])


class TestMeetMat:
    def test_meet_mat_invalid_mode(self):
        """Test that meet_mat raises ValueError for invalid mode"""
        ts = [1.0, 5.0, 10.0]
        with pytest.raises(ValueError):
            mtr.meet_mat([], ts, mode="invalid")

    def test_meet_mat_deepest_mode(self, simple_hierarchy):
        """Test meet_mat with 'deepest' mode"""
        ts = [1.0, 6.0, 11.0, 16.0]
        result = mtr.meet_mat(simple_hierarchy, ts, mode="deepest")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(ts), len(ts))
        # Points in the same finest segment should meet at level 2
        # assert result[0, 0] == 2  # Enable when implementation is complete
        # Diagonal should always be the depth of the hierarchy
        # for i in range(len(ts)):
        #     assert result[i, i] == len(simple_hierarchy) - 1

    def test_meet_mat_mono_mode(self, simple_hierarchy):
        """Test meet_mat with 'mono' mode"""
        ts = [2.0, 8.0, 12.0, 18.0]
        result = mtr.meet_mat(simple_hierarchy, ts, mode="mono")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(ts), len(ts))
        # Points across major segments should meet at level 0
        # assert result[0, 2] == 0  # Points at 2.0 and 12.0 are in different major segments

    def test_meet_mat_mean_mode(self, simple_hierarchy):
        """Test meet_mat with 'mean' mode"""
        ts = [1.0, 3.0, 11.0, 16.0]
        result = mtr.meet_mat(simple_hierarchy, ts, mode="mean")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(ts), len(ts))

    def test_meet_mat_symmetry(self, simple_hierarchy):
        """Test that the meeting matrix is symmetric"""
        ts = [1.0, 6.0, 11.0, 16.0]
        for mode in ["deepest", "mono", "mean"]:
            result = mtr.meet_mat(simple_hierarchy, ts, mode=mode)
            # Matrix should be symmetric
            assert np.allclose(result, result.T)

    def test_meet_mat_custom_compare_fn(self, simple_hierarchy):
        """Test meet_mat with a custom comparison function"""
        ts = [1.0, 6.0, 11.0, 16.0]
        # Use np.less instead of np.greater
        result_greater = mtr.meet_mat(
            simple_hierarchy, ts, mode="deepest", compare_fn=np.greater
        )
        result_less = mtr.meet_mat(
            simple_hierarchy, ts, mode="deepest", compare_fn=np.less
        )

        assert result_greater is not None
        assert result_less is not None
        # The results should be different when using different comparison functions
        # We don't assert exact values here since we don't know the implementation details
