import pytest
import numpy as np
from bnl.core import H, S, multi2H
from bnl.formatting import mireval2multi


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

    hier1 = H([ITVLS1, ITVLS2], [LABELS1, LABELS2])
    hier2 = H([ITVLS3, ITVLS4, ITVLS5], [LABELS3, LABELS4, LABELS5])
    hier3 = H(
        [ITVLS1, ITVLS2, ITVLS3, ITVLS4, ITVLS5],
        [LABELS1, LABELS2, LABELS3, LABELS4, LABELS5],
    )
    return dict(h1=hier1, h2=hier2, h3=hier3)


def test_flat_segmentation_initialization():
    single_segment = np.array([(0, 5)])
    seg = S(single_segment)

    assert np.allclose(seg.itvls, single_segment)
    print(str(single_segment))
    print(seg.Lstar)
    assert str(seg.Lstar[0]) == str(single_segment[0])
    assert seg.T0 == 0
    assert seg.T == 5


@pytest.mark.parametrize("text", [True, False])
def test_flat_segmentation_ploting_parametrized(text, hierarchies):
    fig = hierarchies["h1"].levels[0].plot(text=text)
    # Verify the figure is created
    assert fig is not None


def test_S_L(hierarchies):
    seg = hierarchies["h1"].levels[1]
    assert seg.L(1) == "a"
    assert seg.L(3) == "c"
    assert seg.L(3.5) == "b"


# expected fail with Index Error decoration
@pytest.mark.xfail(raises=IndexError)
def test_S_L_out_of_bounds(hierarchies):
    seg = hierarchies["h1"].levels[1]
    seg.L(10)


def test_S_B(hierarchies):
    seg = hierarchies["h1"].levels[0]
    assert seg.B(0) == 1
    assert seg.B(2.5) == 1
    assert seg.B(1.5) == 0
    assert seg.B(seg.T) == 1
    assert seg.B(seg.T + 1) == 0


def test_S_Bhat(hierarchies):
    # Test if we can get the Bhattacharyya distance
    seg = hierarchies["h1"].levels[0]
    assert len(seg.Bhat([1, 2, 3])) == 3


def test_S_Ahat(hierarchies):
    # Test if we can get the Bhattacharyya distance
    seg = hierarchies["h1"].levels[0]
    assert len(seg.Bhat([1, 2, 3])) == 3
