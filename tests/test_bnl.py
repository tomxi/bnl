import pytest
import numpy as np
from bnl.core import S, H
from bnl.formatting import multi2hier, hier2multi, mireval2multi, multi2mireval


def test_flat_segmentation_initialization():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ["A", "B", "C"]
    seg = S(itvls, labels)

    assert seg.itvls == itvls
    assert seg.labels == labels
    assert seg.Lstar == {0: "A", 1: "B", 2: "C"}
    assert seg.T0 == 0
    assert seg.T == 3


def test_update_bw():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ["A", "B", "C"]
    seg = S(itvls, labels)
    seg.update_bw(2)

    assert seg.Bhat_bw == 2


def test_update_sr():
    pass


def test_label_retrieval():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ["A", "B", "C"]
    seg = S(itvls, labels)

    assert seg.L(0.5) == "A"
    assert seg.L(1.5) == "B"
    assert seg.L(2.5) == "C"


def test_boundary_check():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ["A", "B", "C"]
    seg = S(itvls, labels)

    assert seg.B(1) == 1
    assert seg.B(0.5) == 0


def test_boundary_salience_curve():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ["A", "B", "C"]
    seg = S(itvls, labels)
    seg.update_bw(1)

    bsc = seg.Bhat()
    assert len(bsc) == len(seg.ticks)


def test_label_agreement_indicator():
    pass


def test_hierarchical_segmentation_initialization():
    pass


def test_multi2hier_conversion():
    pass


def test_hier2multi_conversion():
    pass


def test_mireval2multi_conversion():
    pass


def test_multi2mireval_conversion():
    pass
