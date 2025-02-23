import pytest
import numpy as np
from bnl.core import S, H


def test_flat_segmentation_initialization():
    itvls = [[0, 1], [1, 2], [2, 3]]
    labels = ["A", "B", "C"]
    seg = S(itvls, labels)

    assert seg.itvls == itvls
    assert seg.labels == labels
    assert seg.Lstar == {0: "A", 1: "B", 2: "C"}
    assert seg.T0 == 0
    assert seg.T == 3
