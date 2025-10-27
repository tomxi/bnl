import frameless_eval as fle
import pandas as pd

from .metrics import bmeasure2


def relevance(ref, ests, metric="b", debug=False):
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "b30", "b05", "t", "l", "l-mono-lam", "l-mono-1layer"
    # Let's do the simple case of having existing metrics to compute relevance.
    rel = dict()

    # For B-measure and T-measure, we need to make sure that they are monotonic boundaries.
    if metric[0] == "b":
        ref_bc = ref.contour("depth").level("unique")
        if debug:
            ref_bc.plot().show()
        for key, est in ests.items():
            est_bc = (
                est.align(ref).contour("prob").clean("kde", bw=0.5).level("mean_shift", bw=0.15)
            )
            window = int(metric[1:]) * 0.1
            rel[key] = bmeasure2(ref_bc, est_bc, window=window)[2]  # Take the f-score
            if debug:
                est_bc.plot().show()

    # For L-measure, we need expanded labeling... and no monotonicity requirement for now.
    elif metric == "l":
        ref = ref.expand_labels()
        if debug:
            ref.plot().show()
        for key, est in ests.items():
            est = est.align(ref)
            if debug:
                est.plot().show()
            rel[key] = fle.lmeasure(ref.itvls, ref.labels, est.itvls, est.labels)[2]

    # L-measure on monotonicity casted and label decoding.
    elif metric[:6] == "l-mono":
        ref = ref.expand_labels()
        if debug:
            ref.plot().show()
        for key, est in ests.items():
            est = est.align(ref)
            labeling_strat = metric[7:]
            mono_est = (
                est.contour("prob")
                .clean("kde", bw=0.5)
                .level("mean_shift", bw=0.15)
                .to_ms(strategy=labeling_strat, ref_ms=est)
            )
            if debug:
                mono_est.plot().show()
            rel[key] = fle.lmeasure(ref.itvls, ref.labels, mono_est.itvls, mono_est.labels)[2]

    # For T-measure, we need unique leveling, and monotonic casting.
    elif metric == "t":
        ref_ms = ref.contour("depth").level("unique").to_ms()
        if debug:
            ref_ms.plot().show()
        for key, est in ests.items():
            est_ms = (
                est.align(ref)
                .contour("prob")
                .clean("kde", bw=0.5)
                .level("mean_shift", bw=0.15)
                .to_ms()
            )
            rel[key] = fle.lmeasure(ref_ms.itvls, ref_ms.labels, est_ms.itvls, est_ms.labels)[2]
            if debug:
                est_ms.plot().show()

    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return pd.Series(rel, name=metric)


def relevance_per_level(ref, ests, metric="bpc_combo", debug=False):
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "hr", "v", "bpc_combo", "lam_combo"
    if metric == "bpc_combo":
        # Setup est_bpcs and ref_bpc.
        ref_bpc, grid_times = ref.prominence_mat(bw=0.8)
        est_bpcs = {key: est.prominence_mat(bw=0.8)[0].sum(axis=0) for key, est in ests.items()}
        return ref_bpc.sum(axis=0), est_bpcs, grid_times
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return None
