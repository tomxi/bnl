import mir_eval
import numpy as np
from bnl import formatting as fmt

# intervals and labels we are going to use for testing in single layer mireval format
ITVLS1 = np.array([[0, 2.5], [2.5, 5]])
LABELS1 = ["A", "B"]
ITVLS2 = np.array([[0, 1], [1, 2.5], [2.5, 3.5], [3.5, 5]])
LABELS2 = ["a", "b", "c", "b"]

ITVLS3 = np.array([[0, 1], [1, 4], [4, 5]])
LABELS3 = ["Do", "Sol", "Do"]
ITVLS4 = np.array([[0, 1], [1, 3], [3, 4], [4, 5]])
LABELS4 = ["T", "PD", "D", "T"]
ITVLS5 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
LABELS5 = ["I", "IV", "ii", "V", "I"]


def test_mireval_hier_conversion():
    # Test if we can go from mireval format to hierarchical format and back
    hier = fmt.mireval2hier([ITVLS1, ITVLS2], [LABELS1, LABELS2])
    assert len(hier) == 2
    assert hier[0] == [ITVLS1, LABELS1]
    assert hier[1] == [ITVLS2, LABELS2]

    itvls_list, labels_list = fmt.hier2mireval(hier)
    mir_eval.hierarchy.validate_hier_intervals(itvls_list)
    assert itvls_list == [ITVLS1, ITVLS2]
    assert labels_list == [LABELS1, LABELS2]


def test_mireval_multiseg_conversion():
    # Test if we can go from mireval format to jams format and back
    multi_anno = fmt.mireval2multi([ITVLS3, ITVLS4], [LABELS3, LABELS4])
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 5
    itvls, labels = fmt.multi2mireval(multi_anno)
    assert np.allclose(itvls[0], ITVLS3)
    assert np.allclose(itvls[1], ITVLS4)
    assert labels == [LABELS3, LABELS4]


def test_hier_multiseg_conversion():
    # Test if we can go from hier format to jams format and back
    hier = [[ITVLS3, LABELS3], [ITVLS4, LABELS4], [ITVLS5, LABELS5]]
    multi_anno = fmt.hier2multi(hier)
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 5

    hier_back = fmt.multi2hier(multi_anno)
    for l in range(len(hier)):
        assert np.allclose(hier_back[l][0], hier[l][0])
        assert hier_back[l][1] == hier[l][1]


def test_mireval_openseg_conversion():
    # Test if we can go from openseg format to mireval format and back
    openseg_anno = fmt.mirevalflat2openseg(ITVLS2, LABELS2)
    assert openseg_anno.namespace == "segment_open"

    itvls, labels = fmt.openseg2mirevalflat(openseg_anno)
    assert np.allclose(itvls, ITVLS2)
    assert labels == LABELS2


def test_openseg_multiseg_conversion():
    # Test if we can go from openseg format to multi format and back
    openseg_annos = [
        fmt.mirevalflat2openseg(ITVLS1, LABELS1),
        fmt.mirevalflat2openseg(ITVLS2, LABELS2),
    ]
    multi_anno = fmt.openseg2multi(openseg_annos)
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 5

    for lvl in range(2):
        openseg_back = fmt.multi2openseg(multi_anno, layer=lvl)
        assert openseg_back.namespace == "segment_open"
        assert np.allclose(openseg_back.time, openseg_annos[lvl].time)
        assert np.allclose(openseg_back.duration, openseg_annos[lvl].duration)
        assert [obs.value for obs in openseg_back.data] == [
            obs.value for obs in openseg_annos[lvl].data
        ]
