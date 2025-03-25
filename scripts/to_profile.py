import bnl
from bnl import fio, mtr, smtr, fmtr
import numpy as np
import mir_eval.hierarchy as meh
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")
tid = "1437"
est = fio.adobe_hiers(tid)
ref = fio.salami_ref_hiers(tid)[0]
hr, he = mtr.align_hier(ref, est)
mir_eval_result = meh.lmeasure(hr.itvls, hr.labels, he.itvls, he.labels, frame_size=0.2)
bnl_result = mtr.lmeasure(hr, he)
# smtr_result = smtr.lmeasure(hr.itvls, hr.labels, he.itvls, he.labels)
fmtr_result = fmtr.lmeasure(hr.itvls, hr.labels, he.itvls, he.labels)

print(f"Track {tid} L-measure Difference :")
print(f" meh: \t  {np.array(mir_eval_result).round(4)}")
print(f" mtr: \t  {np.array(bnl_result).round(4)}")
print(f" fmtr: \t  {np.array(fmtr_result).round(4)}")
