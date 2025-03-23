import pytest
import numpy as np
import bnl
from bnl import standalone_metrics as mtr
from tests import hierarchies


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

    hier_itvls = [level0_itvls, level1_itvls, level2_itvls]
    hier_labels = [level0_labels, level1_labels, level2_labels]

    return hier_itvls, hier_labels


class TestLabelsAtT:
    def test_labels_at_t_within_interval(self, simple_hierarchy):
        """Test labels_at_t when t is within an interval"""
        assert mtr.labels_at_t(*simple_hierarchy, 1.0) == ["A", "A1", "A1a"]
        assert mtr.labels_at_t(*simple_hierarchy, 6.0) == ["A", "A2", "A2a"]
        assert mtr.labels_at_t(*simple_hierarchy, 16.0) == ["B", "B2", "B2a"]

    def test_labels_at_t_at_boundary(self, simple_hierarchy):
        """Test labels_at_t when t is at the boundary of an interval"""
        assert mtr.labels_at_t(*simple_hierarchy, 5) == ["A", "A2", "A2a"]
        assert mtr.labels_at_t(*simple_hierarchy, 10.0) == ["B", "B1", "B1a"]

    def test_labels_at_t_outside_range(self, simple_hierarchy):
        """Test labels_at_t when t is outside the range of the intervals"""
        assert mtr.labels_at_t(*simple_hierarchy, 21) == [None, None, None]

    def test_labels_at_t_empty_hierarchy(self):
        """Test labels_at_t with an empty hierarchy"""
        assert mtr.labels_at_t([], [], 0) is None


class TestMeet:
    def test_meet_invalid_mode(self, hierarchies):
        """Test that meet raises ValueError for invalid mode"""
        with pytest.raises(ValueError):
            mtr.meet(
                hierarchies["h3"].itvls,
                hierarchies["h3"].labels,
                1.0,
                3.0,
                mode="invalid",
            )

    @pytest.mark.parametrize("mode", ["deepest", "mono"])
    @pytest.mark.parametrize("u", [1, 3, 5])
    @pytest.mark.parametrize("v", [2, 4, 5])
    def test_meet(self, hierarchies, mode, u, v):
        """Test meet with some different modes"""
        h = hierarchies["h3"]
        # Same segment pairs should meet at the deepest level
        assert mtr.meet(h.itvls, h.labels, u, v, mode=mode) == h.meet(u, v, mode=mode)


class TestRelevanceAtT:
    @pytest.mark.parametrize("mode", ["deepest", "mono"])
    @pytest.mark.parametrize("t", [0, 5, 13, 20])
    def test_relevance_at_t(self, simple_hierarchy, t, mode):
        rel_s = bnl.mtr.get_segment_relevance(
            bnl.H(*simple_hierarchy), t, meet_mode=mode
        )
        rel_itvls, rel_values = mtr.relevance_hierarchy_at_t(
            *simple_hierarchy, t, meet_mode=mode
        )
        assert np.allclose(rel_s.itvls, rel_itvls)
        assert np.allclose(rel_s.labels, rel_values)


class TestTripletRecallAtT:
    @pytest.mark.parametrize("t", [1, 3, 5])
    @pytest.mark.parametrize("meet_mode", ["deepest", "mono"])
    @pytest.mark.parametrize("transitive", [True, False])
    def test_triplet_recall_at_t(self, hierarchies, t, meet_mode, transitive):
        """Test triplet recall at t with different parameters"""
        recall = mtr.triplet_recall_at_t(
            hierarchies["h3"].itvls,
            hierarchies["h3"].labels,
            hierarchies["h2"].itvls,
            hierarchies["h2"].labels,
            t,
            meet_mode=meet_mode,
            transitive=transitive,
        )
        reference_recall = bnl.mtr.recall_at_t(
            hierarchies["h3"],
            hierarchies["h2"],
            t,
            meet_mode=meet_mode,
            transitive=transitive,
        )
        # Check if recall is a float
        assert recall == reference_recall


class TestTripletRecall:
    @pytest.mark.parametrize("meet_mode", ["deepest", "mono"])
    @pytest.mark.parametrize("transitive", [True, False])
    def test_triplet_recall_different_hierarchies(
        self, hierarchies, meet_mode, transitive
    ):
        """Test triplet recall with different hierarchies"""
        h3 = hierarchies["h3"]
        h2 = hierarchies["h2"]
        recall = mtr.triplet_recall(
            h3.itvls,
            h3.labels,
            h2.itvls,
            h2.labels,
            meet_mode=meet_mode,
            transitive=transitive,
        )
        ref_recall = bnl.mtr.recall(
            h3,
            h2,
            meet_mode=meet_mode,
            transitive=transitive,
        )
        # Check if recall is a float
        assert isinstance(recall, float)
        assert recall == ref_recall


class TestLMeasure:
    @pytest.mark.parametrize("meet_mode", ["deepest", "mono"])
    def test_lmeasure_different_hierarchies(self, hierarchies, meet_mode):
        """Test L-measure with different hierarchies"""
        h3 = hierarchies["h3"]
        h2 = hierarchies["h2"]
        precision, recall, f1 = mtr.lmeasure(
            h3.itvls,
            h3.labels,
            h2.itvls,
            h2.labels,
            meet_mode=meet_mode,
        )
        ref_precision, ref_recall, ref_f1 = bnl.mtr.lmeasure(
            h3,
            h2,
            meet_mode=meet_mode,
        )
        # Check if recall is a float
        assert isinstance(recall, float)
        assert np.allclose([precision, recall, f1], [ref_precision, ref_recall, ref_f1])


class TestTMeasure:
    @pytest.mark.parametrize("meet_mode", ["deepest", "mono"])
    @pytest.mark.parametrize("transitive", [True, False])
    def test_tmeasure_different_hierarchies(self, hierarchies, meet_mode, transitive):
        """Test T-measure with different hierarchies"""
        h3 = hierarchies["h3"]
        h2 = hierarchies["h2"]
        precision, recall, f1 = mtr.tmeasure(
            h3.itvls,
            h3.labels,
            h2.itvls,
            h2.labels,
            meet_mode=meet_mode,
            transitive=transitive,
        )
        ref_precision, ref_recall, ref_f1 = bnl.mtr.tmeasure(
            h3,
            h2,
            meet_mode=meet_mode,
            transitive=transitive,
        )
        assert isinstance(recall, float)
        assert np.allclose([precision, recall, f1], [ref_precision, ref_recall, ref_f1])


# class TestPairwiseRecall:
#     def test_pairwise_recall_exact_match(self):
#         """Test pairwise recall with exact matching segments"""
#         ref_itvls = [(0, 5), (5, 10)]
#         ref_labels = ["A", "B"]
#         est_itvls = [(0, 5), (5, 10)]
#         est_labels = ["A", "B"]

#         t = 2.5  # middle of first segment
#         recall = mtr.pairwise_recall(ref_itvls, ref_labels, est_itvls, est_labels, t)
#         assert isinstance(recall, float)
#         # Perfect match should give recall of 1.0
#         # assert recall == 1.0  # Enable when implementation is complete

#     def test_pairwise_recall_with_window(self):
#         """Test pairwise recall with window parameter"""
#         ref_itvls = [(0, 5), (5, 10)]
#         ref_labels = ["A", "B"]
#         est_itvls = [(0, 4.8), (4.8, 10)]  # Slight boundary deviation
#         est_labels = ["A", "B"]

#         t = 4.9  # Near the boundary

#         # Without window, the recall might be poor
#         recall_no_window = mtr.pairwise_recall(
#             ref_itvls, ref_labels, est_itvls, est_labels, t, window=0
#         )

#         # With window, recall should improve
#         recall_with_window = mtr.pairwise_recall(
#             ref_itvls, ref_labels, est_itvls, est_labels, t, window=0.5
#         )

#         assert isinstance(recall_no_window, float)
#         assert isinstance(recall_with_window, float)
#         # Window should help in this case
#         # assert recall_with_window >= recall_no_window  # Enable when implementation is complete


# class TestEntropy:
#     def test_entropy_uniform_distribution(self):
#         """Test entropy calculation with uniform distribution"""
#         # Equal-sized segments with different labels
#         itvls = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
#         labels = ["A", "B", "C", "D"]

#         result = mtr.entropy(itvls, labels)
#         assert isinstance(result, float)
#         # Maximum entropy for uniform distribution
#         # assert np.isclose(result, 2.0)  # log2(4) = 2, uncomment when implemented

#     def test_entropy_single_segment(self):
#         """Test entropy calculation with single segment"""
#         itvls = np.array([[0, 5]])
#         labels = ["A"]

#         result = mtr.entropy(itvls, labels)
#         assert isinstance(result, float)
#         assert isinstance(result, float)
#         # Zero entropy for single segment
#         # assert np.isclose(result, 0.0)  # uncomment when implemented

#     def test_entropy_with_predefined_intervals(self):
#         """Test entropy using predefined test intervals"""
#         result = mtr.entropy(tests.ITVLS1, tests.LABELS1)
#         assert isinstance(result, float)

#         result = mtr.entropy(tests.ITVLS5, tests.LABELS5)
#         assert isinstance(result, float)

#     def test_entropy_with_repeated_labels(self):
#         """Test entropy with repeated labels"""
#         # tests.ITVLS2 has repeated 'b' labels
#         result = mtr.entropy(tests.ITVLS2, tests.LABELS2)
#         assert isinstance(result, float)


# class TestConditionalEntropy:
#     def test_conditional_entropy_identical(self):
#         """Test conditional entropy when segmentations are identical"""
#         result = mtr.conditional_entropy(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
#         )
#         assert isinstance(result, float)
#         # Conditional entropy should be 0 for identical segmentations
#         # assert np.isclose(result, 0.0)  # uncomment when implemented

#     def test_conditional_entropy_independent(self):
#         """Test conditional entropy with independent segmentations"""
#         result = mtr.conditional_entropy(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS3, tests.LABELS3
#         )
#         assert isinstance(result, float)

#     def test_conditional_entropy_refined(self):
#         """Test conditional entropy with one segmentation refining another"""
#         # tests.ITVLS2 refines tests.ITVLS1
#         result = mtr.conditional_entropy(
#             tests.ITVLS2, tests.LABELS2, tests.ITVLS1, tests.LABELS1
#         )
#         assert isinstance(result, float)


# class TestVmeasure:
#     def test_vmeasure_identical(self):
#         """Test V-measure with identical segmentations"""
#         precision, recall, f1 = mtr.vmeasure(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])
#         # Perfect match should give values of 1.0
#         # assert np.isclose(precision, 1.0)  # uncomment when implemented
#         # assert np.isclose(recall, 1.0)
#         # assert np.isclose(f1, 1.0)

#     def test_vmeasure_refined(self):
#         """Test V-measure with one segmentation refining another"""
#         precision, recall, f1 = mtr.vmeasure(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS2, tests.LABELS2
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])

#     def test_vmeasure_different_structures(self):
#         """Test V-measure with completely different structures"""
#         precision, recall, f1 = mtr.vmeasure(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS3, tests.LABELS3
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])

#     def test_vmeasure_edge_cases(self):
#         """Test V-measure with edge cases"""
#         # Single segment
#         single_itvls = np.array([[0, 5]])
#         single_labels = ["A"]

#         precision, recall, f1 = mtr.vmeasure(
#             single_itvls, single_labels, tests.ITVLS1, tests.LABELS1
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])

#         # Different lengths
#         precision, recall, f1 = mtr.vmeasure(
#             tests.ITVLS5, tests.LABELS5, tests.ITVLS1, tests.LABELS1
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])


# class TestPairClustering:
#     def test_pair_clustering_identical(self):
#         """Test pair clustering with identical segmentations"""
#         precision, recall, f1 = mtr.pair_clustering(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS1, tests.LABELS1
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])
#         # Perfect match should give values of 1.0
#         # assert np.isclose(precision, 1.0)  # uncomment when implemented
#         # assert np.isclose(recall, 1.0)
#         # assert np.isclose(f1, 1.0)

#     def test_pair_clustering_with_refinement(self):
#         """Test pair clustering with one segmentation refining another"""
#         precision, recall, f1 = mtr.pair_clustering(
#             tests.ITVLS1, tests.LABELS1, tests.ITVLS2, tests.LABELS2
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])

#         # Reversed
#         precision_rev, recall_rev, f1_rev = mtr.pair_clustering(
#             tests.ITVLS2, tests.LABELS2, tests.ITVLS1, tests.LABELS1
#         )
#         assert all(isinstance(x, float) for x in [precision_rev, recall_rev, f1_rev])

#     def test_pair_clustering_different_structures(self):
#         """Test pair clustering with completely different structures"""
#         precision, recall, f1 = mtr.pair_clustering(
#             tests.ITVLS3, tests.LABELS3, tests.ITVLS4, tests.LABELS4
#         )
#         assert all(isinstance(x, float) for x in [precision, recall, f1])
