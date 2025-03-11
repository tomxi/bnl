import numpy as np

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

from .test_core import make_hierarchies
