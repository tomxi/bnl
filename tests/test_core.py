import pytest
import numpy as np
from bnl.core import S, H, multi2H, levels2H
from bnl.formatting import mireval2multi, multi2mireval
import tests


def test_flat_segmentation_initialization():
    single_segment = np.array([(0, 5)])
    seg = S(single_segment)

    assert np.allclose(seg.itvls, single_segment)
    print(str(single_segment))
    print(seg.Lstar)
    assert seg.Lstar == {0: str(single_segment[0])}
    assert seg.T0 == 0
    assert seg.T == 5


@pytest.mark.parametrize("legend_ncol", [6, None])
@pytest.mark.parametrize(["y_label", "relabel"], [[True, False], [False, True]])
def test_flat_segmentation_ploting_parametrized(legend_ncol, y_label, relabel):
    seg = S(tests.ITVLS1, tests.LABELS1)
    fig = seg.plot(legend_ncol=legend_ncol, y_label=y_label, relabel=relabel)
    # Verify the figure is created
    assert fig is not None


def test_hierarchical_segmentation_initialization():
    test_anno = mireval2multi(
        [tests.ITVLS1, tests.ITVLS2], [tests.LABELS1, tests.LABELS2]
    )
    hier = multi2H(test_anno)
    assert hier.d == 2


def test_S_update_sr():
    # Test if we can update the sample rate
    seg = S(tests.ITVLS1, tests.LABELS1)

    # update to default again to hit coverage
    seg.update_sr(10)
    num_ticks_sr10 = len(seg.ticks)
    seg.update_sr(2)
    num_ticks_sr2 = len(seg.ticks)

    assert (num_ticks_sr2 - 1) * 5 == num_ticks_sr10 - 1


def test_S_L():
    seg = S(tests.ITVLS2, tests.LABELS2)
    assert seg.L(1) == "b"
    assert seg.L(3) == "c"
    assert seg.L(3.5) == "b"


# expected fail with Index Error decoration
@pytest.mark.xfail(raises=IndexError)
def test_S_L_out_of_bounds():
    seg = S(tests.ITVLS2, tests.LABELS2)
    seg.L(10)


def test_S_B():
    seg = S(tests.ITVLS1, tests.LABELS1)
    assert seg.B(0) == 1
    assert seg.B(2.5) == 1
    assert seg.B(1.5) == 0
    assert seg.B(5) == 1
    assert seg.B(seg.T + 1) == 0


def test_S_Bhat():
    # Test if we can get the Bhattacharyya distance
    seg = S(tests.ITVLS1, tests.LABELS1)
    assert len(seg.Bhat()) > 0
    assert len(seg.Bhat([1, 2, 3])) == 3


def test_S_Ahat():
    # Test if we can get the Bhattacharyya distance
    seg = S(tests.ITVLS1, tests.LABELS1)
    assert len(seg.Ahat()) == len(seg.beta) - 1
    assert seg.Ahat([1, 2, 3]).shape == (2, 2)


def test_H_init():
    pass


def test_H_update_sr():
    pass


def test_H_update_bw():
    pass


def test_H_A():
    pass


def test_H_B():
    pass


def test_H_Ahat():
    pass


def test_H_Bhat():
    pass


def test_H_mono_checks():
    pass


def test_H_M():
    pass


def test_H_Mhat():
    pass


def test_H_plot():
    pass


def test_H_decode_B():
    pass


def test_H_decode_L():
    pass


def test_H_decode_full():
    pass
