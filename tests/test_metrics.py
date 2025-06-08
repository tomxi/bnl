import mir_eval
from mir_eval import hierarchy as meh
import bnl
from bnl import fio, mtr, plotting as viz # Changed viz import
import numpy as np
import tests # This might be an unused import or a local 'tests' package reference?
import warnings
import pytest # Import pytest


@pytest.mark.skip(reason="Requires external data files not available in the testing environment.")
def test_lmeasure_alignment(tid="1437", atol=1e-2, frame_size=0.5):
    warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")

    all_estimations = fio.salami_adobe_hiers(tid) # Corrected function name
    all_references = fio.salami_ref_hiers(tid) # This also returns a dict of Segmentation objects

    for ref_annotator_name, ref_segmentation in all_references.items():
        for est_param_name, est_segmentation in all_estimations.items():
            # ref_segmentation and est_segmentation are Segmentation objects.
            # Their .itvls and .labels attributes should provide the lists of arrays/lists
            # that mtr.align_hier and the metric functions expect.

            ref_itvls_list = ref_segmentation.itvls
            ref_labels_list = ref_segmentation.labels
            est_itvls_list = est_segmentation.itvls
            est_labels_list = est_segmentation.labels

            aligned_ref_itvls, aligned_ref_labels, aligned_est_itvls, aligned_est_labels = mtr.align_hier(
                ref_itvls_list, ref_labels_list, est_itvls_list, est_labels_list
            )

            mir_eval_result = meh.lmeasure(
                aligned_ref_itvls, aligned_ref_labels, aligned_est_itvls, aligned_est_labels, frame_size=frame_size
            )
            bnl_result = mtr.lmeasure(
                aligned_ref_itvls, aligned_ref_labels, aligned_est_itvls, aligned_est_labels
            )
            diff = np.array(mir_eval_result) - np.array(bnl_result)

            print(f"Track {tid}, Ref: {ref_annotator_name}, Est: {est_param_name} L-measure Diff (fs={frame_size}):")
            print(f"      \t  {np.abs(np.array(diff)*100).round(3)}%")
            print(f" meh: \t  {np.array(mir_eval_result).round(4)}")
            print(f" bnl: \t  {np.array(bnl_result).round(4)}")

            assert np.allclose(mir_eval_result, bnl_result, atol=atol)
