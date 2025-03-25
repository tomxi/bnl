import bnl
from bnl import fio, mtr, smtr
import numpy as np
import mir_eval.hierarchy as meh
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")
tid = "1437"
est = fio.adobe_hiers(tid)
ref = fio.salami_ref_hiers(tid)[0]
hr, he = mtr.align_hier(ref, est)
# mir_eval_result = meh.lmeasure(
#     hr.itvls, hr.labels, he.itvls, he.labels, frame_size=frame_size
# )
bnl_result = mtr.lmeasure(hr, he)
smtr_result = smtr.lmeasure(hr.itvls, hr.labels, he.itvls, he.labels)
diff = np.array(smtr_result) - np.array(bnl_result)

print(f"Track {tid} L-measure Difference :")
print(f"      \t  {np.abs(np.array(diff)*100).round(3)}%")
print(f" meh: \t  {np.array(smtr_result).round(4)}")
print(f" bnl: \t  {np.array(bnl_result).round(4)}")
