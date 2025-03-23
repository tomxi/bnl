import pytest
import numpy as np
import bnl
from bnl import standalone_metrics as mtr
from tests import hierarchies
import mir_eval


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


class TestPairwiseRecall:
    def test_pairwise_recall_different_segmentations(self, hierarchies):
        """Test pairwise recall with different segmentations"""
        s1 = hierarchies["h3"].levels[1]
        s2 = hierarchies["h3"].levels[4]
        recall = mtr.pairwise_recall(
            s1.itvls,
            s1.labels,
            s2.itvls,
            s2.labels,
        )
        ref_pairwise = mir_eval.segment.pairwise(
            s1.itvls, s1.labels, s2.itvls, s2.labels, frame_size=0.01
        )
        ref_precision, ref_recall, _ = ref_pairwise
        assert ref_precision >= 0.0
        assert np.allclose(recall, ref_recall, atol=0.005)

    def test_pairwise_recall_empty_segmentations(self):
        """Test pairwise recall with empty segmentations"""
        recall = mtr.pairwise_recall([], [], [], [])
        assert np.isnan(recall)

    def test_pairwise_recall_single_interval(self):
        """Test pairwise recall with single interval segmentations"""
        ref_itvls = [(0, 10)]
        ref_labels = ["A"]
        est_itvls = [(0, 10)]
        est_labels = ["B"]
        recall = mtr.pairwise_recall(ref_itvls, ref_labels, est_itvls, est_labels)
        assert np.allclose(recall, 1.0)


class TestEntropy:
    def test_entropy(self, simple_hierarchy):
        """Test entropy calculation"""
        itvls, labels = simple_hierarchy
        # Test with a single interval
        assert mtr.entropy([[0, 10]], ["a"]) == 0
        # Test with multiple intervals
        assert mtr.entropy(itvls[1], labels[1]) > 0.0

    def test_entropy_empty(self):
        """Test entropy with empty input"""
        assert mtr.entropy([], []) == 0.0


class TestVmeasure:
    def test_vmeasure(self, hierarchies):
        """Test V-measure with different hierarchies"""
        s3 = hierarchies["h3"].levels[3]
        s2 = hierarchies["h2"].levels[2]
        v_measure = mtr.vmeasure(
            s3.itvls,
            s3.labels,
            s2.itvls,
            s2.labels,
        )
        ref_v_measure = mir_eval.segment.vmeasure(
            s3.itvls,
            s3.labels,
            s2.itvls,
            s2.labels,
            frame_size=0.1,
        )
        assert np.allclose(v_measure, ref_v_measure)


class TestConditionalEntropy:
    def test_conditional_entropy(self, simple_hierarchy):
        """Test conditional entropy calculation"""
        itvls, labels = simple_hierarchy
        # Test with a single interval
        assert mtr.conditional_entropy([[0, 10]], ["a"], [[0, 10]], ["b"]) == 0
        # Test with multiple intervals
        assert mtr.conditional_entropy(itvls[1], labels[1], itvls[0], labels[0]) >= 0.0

    def test_conditional_entropy_different_segmentations(self, hierarchies):
        """Test conditional entropy with different segmentations"""
        s1 = hierarchies["h3"].levels[1]
        s2 = hierarchies["h3"].levels[4]
        cond_entropy = mtr.conditional_entropy(
            s1.itvls,
            s1.labels,
            s2.itvls,
            s2.labels,
        )
        assert cond_entropy >= 0.0
