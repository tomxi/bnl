import pytest
import numpy as np
from fwx.fwx import S, H
from fwx.formatting import multi2hier, hier2multi, mireval2multi
import jams

def test_flat_segmentation_initialization():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)

    assert seg.itvls == itvls
    assert seg.labels == labels
    assert seg.Lstar == {0: 'A', 1: 'B', 2: 'C'}
    assert seg.T0 == 0
    assert seg.T == 3

def test_update_bw():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)
    seg.update_bw(2)

    assert seg.Bhat_bw == 2

def test_update_sr():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)
    seg.update_sr(20)

    assert seg.sr == 20
    assert len(seg.ticks) == 21

def test_label_retrieval():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)

    assert seg.L(0.5) == 'A'
    assert seg.L(1.5) == 'B'
    assert seg.L(2.5) == 'C'

def test_boundary_check():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)

    assert seg.B(1) == 1
    assert seg.B(0.5) == 0

def test_boundary_salience_curve():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)
    seg.update_bw(1)

    bsc = seg.Bhat()
    assert len(bsc) == len(seg.ticks)

def test_label_agreement_indicator():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ['A', 'B', 'C']
    seg = S(itvls, labels)

    agreement = seg.A(bs=[0, 1, 2, 3])
    assert agreement.shape == (3, 3)

def test_hierarchical_segmentation_initialization():
    itvls = [[0, 1], [1, 2]]
    labels = ['A', 'B']
    hier = H(itvls, labels)

    assert hier.itvls == itvls
    assert hier.labels == labels

def test_multi2hier_conversion():
    itvls = [[0, 1], [1, 2]]
    labels = ['A', 'B']
    anno = mireval2multi(itvls, labels)
    hier = multi2hier(anno)

    assert len(hier) == 2  # Check number of levels

def test_hier2multi_conversion():
    itvls = [[[0, 1]], [[1, 2]]]
    labels = [['A'], ['B']]
    hier = [itvls, labels]
    anno = hier2multi(hier)

    assert len(anno.data) == 2  # Check number of annotations

def test_mireval2multi_conversion():
    itvls = np.array([[0, 1], [1, 2]])
    labels = ['A', 'B']
    anno = mireval2multi(itvls, labels)

    assert len(anno.data) == 2  # Check number of annotations

def test_multi2mireval_conversion():
    itvls = np.array([[0, 1], [1, 2]])
    labels = ['A', 'B']
    anno = mireval2multi(itvls, labels)
    intervals, lbls = multi2mireval(anno)

    assert len(intervals) == 2  # Check number of intervals
    assert lbls == labels  # Check labels match