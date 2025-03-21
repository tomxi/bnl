import mir_eval
from mir_eval import hierarchy as meh
import bnl
from bnl import fio, mtr, viz
import numpy as np
import tests
import warnings


def test_lmeasure_alignment(tid="1437", atol=1e-2, frame_size=0.5, verbose=False):
    warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")
    est = fio.adobe_hiers(tid)
    refs = fio.salami_ref_hiers(tid)
    for ref in refs:
        hr, he = mtr.align_hier(ref, est)
        mir_eval_result = meh.lmeasure(
            hr.itvls, hr.labels, he.itvls, he.labels, frame_size=frame_size
        )
        bnl_result = mtr.lmeasure(hr, he)
        diff = np.array(mir_eval_result) - np.array(bnl_result)

        if verbose:
            print(f"Track {tid} L-measure Difference (fs={frame_size}):")
            print(f"      \t  {np.abs(np.array(diff)*100).round(3)}%")
            print(f" meh: \t  {np.array(mir_eval_result).round(4)}")
            print(f" bnl: \t  {np.array(bnl_result).round(4)}")

        assert np.allclose(mir_eval_result, bnl_result, atol=atol)
