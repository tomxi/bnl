import mir_eval
import numpy as np
from bnl import formatting as fmt
import tests


def test_mireval_hier_conversion():
    # Test if we can go from mireval format to hierarchical format and back
    hier = fmt.mireval2hier(
        [tests.ITVLS1, tests.ITVLS2], [tests.LABELS1, tests.LABELS2]
    )
    assert len(hier) == 2
    assert hier[0] == [tests.ITVLS1, tests.LABELS1]
    assert hier[1] == [tests.ITVLS2, tests.LABELS2]

    itvls_list, labels_list = fmt.hier2mireval(hier)
    mir_eval.hierarchy.validate_hier_intervals(itvls_list)
    assert itvls_list == [tests.ITVLS1, tests.ITVLS2]
    assert labels_list == [tests.LABELS1, tests.LABELS2]


def test_mireval_multiseg_conversion():
    # Test if we can go from mireval format to jams format and back
    multi_anno = fmt.mireval2multi(
        [tests.ITVLS3, tests.ITVLS4], [tests.LABELS3, tests.LABELS4]
    )
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 6.01
    itvls, labels = fmt.multi2mireval(multi_anno)
    assert np.allclose(itvls[0], tests.ITVLS3)
    assert np.allclose(itvls[1], tests.ITVLS4)
    assert labels == [tests.LABELS3, tests.LABELS4]


def test_hier_multiseg_conversion():
    # Test if we can go from hier format to jams format and back
    hier = [
        [tests.ITVLS3, tests.LABELS3],
        [tests.ITVLS4, tests.LABELS4],
        [tests.ITVLS5, tests.LABELS5],
    ]
    multi_anno = fmt.hier2multi(hier)
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 6.01

    hier_back = fmt.multi2hier(multi_anno)
    for l in range(len(hier)):
        assert np.allclose(hier_back[l][0], hier[l][0])
        assert hier_back[l][1] == hier[l][1]


def test_mireval_openseg_conversion():
    # Test if we can go from openseg format to mireval format and back
    openseg_anno = fmt.mirevalflat2openseg(tests.ITVLS2, tests.LABELS2)
    assert openseg_anno.namespace == "segment_open"

    itvls, labels = fmt.openseg2mirevalflat(openseg_anno)
    assert np.allclose(itvls, tests.ITVLS2)
    assert labels == tests.LABELS2


def test_openseg_multiseg_conversion():
    # Test if we can go from openseg format to multi format and back
    openseg_annos = [
        fmt.mirevalflat2openseg(tests.ITVLS1, tests.LABELS1),
        fmt.mirevalflat2openseg(tests.ITVLS2, tests.LABELS2),
    ]
    multi_anno = fmt.openseg2multi(openseg_annos)
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 6.01

    for lvl in range(2):
        openseg_back = fmt.multi2openseg(multi_anno, layer=lvl)
        assert openseg_back.namespace == "segment_open"
        assert np.allclose(openseg_back.time, openseg_annos[lvl].time)
        assert np.allclose(openseg_back.duration, openseg_annos[lvl].duration)
        assert [obs.value for obs in openseg_back.data] == [
            obs.value for obs in openseg_annos[lvl].data
        ]
