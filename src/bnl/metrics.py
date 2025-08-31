import mir_eval
import numpy as np
import pandas as pd

from .core import BoundaryContour as BC


def safe_div(numerator, denominator, default=0):
    if denominator == 0:
        return default
    return numerator / float(denominator)


def bmeasure(
    ref_bc: BC,
    est_bc: BC,
    window: float = 0.5,
    trim: bool = False,
    reduced: bool = False,
    raw: bool = False,
    f_beta: float = 1,
):
    # Get matching boundaries
    if trim:
        ref_bs = ref_bc.bs[1:-1]
        est_bs = est_bc.bs[1:-1]
    else:
        ref_bs = ref_bc.bs
        est_bs = est_bc.bs
    ref_b_times = [b.time for b in ref_bs]
    est_b_times = [b.time for b in est_bs]
    match_idx = mir_eval.util.match_events(ref_b_times, est_b_times, window)

    # Get P(t|H) and P(t_hat|H_hat) using (t, t_hat) in matching boundaries
    rec_ref_prom = []
    rec_est_prom = []
    prec_ref_prom = []
    prec_est_prom = []

    if reduced:
        # only consider matched boundaries
        for ref_b_idx, est_b_idx in match_idx:
            rec_ref_prom.append(ref_bs[ref_b_idx].salience)
            rec_est_prom.append(est_bs[est_b_idx].salience)
        prec_ref_prom = rec_ref_prom.copy()
        prec_est_prom = rec_est_prom.copy()
    else:
        # consider all boundaries in reference, and insert 0 prominence for unmatched est bdry
        ref_to_est_idx_map = {pair[0]: pair[1] for pair in match_idx}
        for ref_b_idx, ref_b in enumerate(ref_bs):
            rec_ref_prom.append(ref_b.salience)
            if ref_b_idx in ref_to_est_idx_map:
                est_b_idx = ref_to_est_idx_map[ref_b_idx]
                rec_est_prom.append(est_bs[est_b_idx].salience)
            else:
                rec_est_prom.append(0)

        # Now do the same for precision using boundaries in estimate
        est_to_ref_idx_map = {pair[1]: pair[0] for pair in match_idx}
        for est_b_idx, est_b in enumerate(est_bs):
            prec_est_prom.append(est_b.salience)
            if est_b_idx in est_to_ref_idx_map:
                ref_b_idx = est_to_ref_idx_map[est_b_idx]
                prec_ref_prom.append(ref_bs[ref_b_idx].salience)
            else:
                prec_ref_prom.append(0)

    rec_ref_prom = np.array(rec_ref_prom)
    rec_est_prom = np.array(rec_est_prom)
    prec_ref_prom = np.array(prec_ref_prom)
    prec_est_prom = np.array(prec_est_prom)

    # Use mir_eval.hierarchy ranking comparison
    recall_inv, recall_norm = mir_eval.hierarchy._compare_frame_rankings(
        rec_ref_prom, rec_est_prom, transitive=True
    )
    precision_inv, precision_norm = mir_eval.hierarchy._compare_frame_rankings(
        prec_est_prom, prec_ref_prom, transitive=True
    )

    raw_data = {
        "po_precision_num": float(precision_norm - precision_inv),
        "po_precision_denom": float(precision_norm),
        "po_recall_num": float(recall_norm - recall_inv),
        "po_recall_denom": float(recall_norm),
        "hr_num": float(len(match_idx)),
        "hr_precision_denom": float(len(est_bs)),
        "hr_recall_denom": float(len(ref_bs)),
        "rec_ref_prom": rec_ref_prom,
        "rec_est_prom": rec_est_prom,
        "prec_ref_prom": prec_ref_prom,
        "prec_est_prom": prec_est_prom,
    }

    if raw:
        return raw_data
    else:
        po_recall = safe_div(raw_data["po_recall_num"], raw_data["po_recall_denom"], 1)
        po_precision = safe_div(raw_data["po_precision_num"], raw_data["po_precision_denom"], 1)
        hr_recall = safe_div(raw_data["hr_num"], raw_data["hr_recall_denom"], 1)
        hr_precision = safe_div(raw_data["hr_num"], raw_data["hr_precision_denom"], 1)
        return {
            "po_p": po_precision,
            "po_r": po_recall,
            "po_f": mir_eval.util.f_measure(po_precision, po_recall, beta=f_beta),
            "hr_p": hr_precision,
            "hr_r": hr_recall,
            "hr_f": mir_eval.util.f_measure(hr_precision, hr_recall, beta=f_beta),
        }


def bmeasure_suite(
    ref_bc: BC, est_bc: BC, track_id: str = "tid", trim: bool = True, windows: list = (0.5, 3)
):
    records = []
    for reduced in [True, False]:
        for window in windows:
            scores = bmeasure(ref_bc, est_bc, window=window, reduced=reduced, trim=trim)
            for metric, score in scores.items():
                records.append(
                    {
                        "track_id": track_id,
                        "metric": metric.split("_")[0],
                        "prf": metric.split("_")[1],
                        "reduced": reduced,
                        "window": window,
                        "score": float(score),
                    }
                )

    df = pd.DataFrame(records)
    return (
        df.assign(
            metric=lambda s: s["metric"].mask(
                s["metric"].eq("po"), s["reduced"].map({True: "po", False: "po_exp"})
            )
        )
        .drop(columns="reduced")
        .drop_duplicates()
        .sort_values(by=["track_id", "window", "prf", "metric"])
    )
