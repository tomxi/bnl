import mir_eval
import numpy as np
import pytest
from bnl import formatting as fmt


@pytest.fixture(scope="module")
def test_data():
    """Create all test intervals and labels in one fixture."""
    return {
        "itvls1": np.array([[0, 2.5], [2.5, 6.01]]),
        "labels1": ["A", "B"],
        "itvls2": np.array([[0, 1.2], [1.2, 2.5], [2.5, 3.5], [3.5, 6.01]]),
        "labels2": ["a", "b", "c", "b"],
        "itvls3": np.array([[0, 1.2], [1.2, 4], [4, 6.01]]),
        "labels3": ["Mi", "Re", "Do"],
        "itvls4": np.array(
            [[0, 0.8], [0.8, 1.6], [1.6, 2.4], [2.4, 3.2], [3.2, 4.0], [4.0, 6.01]]
        ),
        "labels4": ["do", "re", "mi", "fa", "sol", "la"],
        "itvls5": np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6.01]]),
        "labels5": ["1", "2", "3", "4", "5", "6"],
    }


def test_mireval_multiseg_conversion(test_data):
    # Test if we can go from mireval format to jams format and back
    multi_anno = fmt.mireval2multi(
        [test_data["itvls3"], test_data["itvls4"]],
        [test_data["labels3"], test_data["labels4"]],
    )
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 6.01
    itvls, labels = fmt.multi2mireval(multi_anno)
    assert np.allclose(itvls[0], test_data["itvls3"])
    assert np.allclose(itvls[1], test_data["itvls4"])
    assert labels[0] == test_data["labels3"]
    assert labels[1] == test_data["labels4"]


def test_hier_multiseg_conversion(test_data):
    # Test if we can go from hier format to jams format and back
    hier = [
        [test_data["itvls3"], test_data["labels3"]],
        [test_data["itvls4"], test_data["labels4"]],
        [test_data["itvls5"], test_data["labels5"]],
    ]
    multi_anno = fmt.hier2multi(hier)
    assert multi_anno.namespace == "multi_segment"
    assert multi_anno.duration == 6.01

    hier_back = fmt.multi2hier(multi_anno)
    for l in range(len(hier)):
        assert np.allclose(hier_back[l][0], hier[l][0])
        assert hier_back[l][1] == hier[l][1]


def test_mireval_openseg_conversion(test_data):
    # Test if we can go from openseg format to mireval format and back
    openseg_anno = fmt.mirevalflat2openseg(test_data["itvls2"], test_data["labels2"])
    assert openseg_anno.namespace == "segment_open"

    itvls, labels = fmt.openseg2mirevalflat(openseg_anno)
    assert np.allclose(itvls, test_data["itvls2"])
    assert labels == test_data["labels2"]


def test_openseg_multiseg_conversion(test_data):
    # Test if we can go from openseg format to multi format and back
    openseg_annos = [
        fmt.mirevalflat2openseg(test_data["itvls1"], test_data["labels1"]),
        fmt.mirevalflat2openseg(test_data["itvls2"], test_data["labels2"]),
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
