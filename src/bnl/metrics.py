import mir_eval
import numpy as np
import pandas as pd
from mir_eval.util import f_measure

from .core import BoundaryContour as BC


def safe_div(numerator, denominator, default=0):
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def harm_mean(a, b, alpha=0.5):
    assert alpha <= 1 and alpha >= 0
    return safe_div(a * b, (1 - alpha) * a + alpha * b, 0)


def bmeasure(
    ref_bc: BC,
    est_bc: BC,
    window: float = 0.5,
    matched_poa: bool = False,
    raw: bool = False,
    f_beta: float = 1,
):
    # Get matching boundaries
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

    if matched_poa:
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
    rec_inv, rec_norm = mir_eval.hierarchy._compare_frame_rankings(
        rec_ref_prom, rec_est_prom, transitive=True
    )
    rec_total_pair = (len(rec_ref_prom) ** 2 - len(rec_ref_prom)) / 2.0

    prec_inv, prec_norm = mir_eval.hierarchy._compare_frame_rankings(
        prec_est_prom, prec_ref_prom, transitive=True
    )
    prec_total_pair = (len(prec_est_prom) ** 2 - len(prec_est_prom)) / 2.0

    raw_data = {
        "prec_poa_num": float(prec_norm - prec_inv),
        "prec_poa_denom": float(prec_norm),
        "est_psr": safe_div(prec_norm, prec_total_pair),
        "rec_poa_num": float(rec_norm - rec_inv),
        "rec_poa_denom": float(rec_norm),
        "ref_psr": safe_div(rec_norm, rec_total_pair),
        "matched_bdry": float(len(match_idx)),
        "est_bdry": float(len(est_bs)),
        "ref_bdry": float(len(ref_bs)),
        "rec_ref_prom": rec_ref_prom,
        "rec_est_prom": rec_est_prom,
        "prec_ref_prom": prec_ref_prom,
        "prec_est_prom": prec_est_prom,
    }

    if raw:
        return raw_data
    else:
        poa_recall = safe_div(raw_data["rec_poa_num"], raw_data["rec_poa_denom"], 1)
        poa_precision = safe_div(raw_data["prec_poa_num"], raw_data["prec_poa_denom"], 1)
        hr_recall = safe_div(raw_data["matched_bdry"], raw_data["ref_bdry"], 1)
        hr_precision = safe_div(raw_data["matched_bdry"], raw_data["est_bdry"], 1)
        b_precision = f_measure(hr_precision, poa_precision, beta=raw_data["est_psr"] ** 0.5)
        b_recall = f_measure(hr_recall, poa_recall, beta=raw_data["ref_psr"] ** 0.5)
        return {
            "poa_p": poa_precision,
            "poa_r": poa_recall,
            "poa_f": f_measure(poa_precision, poa_recall, beta=f_beta),
            "hr_p": hr_precision,
            "hr_r": hr_recall,
            "hr_f": f_measure(hr_precision, hr_recall, beta=f_beta),
            "b_p": b_precision,
            "b_r": b_recall,
            "b_f": f_measure(b_precision, b_recall, beta=f_beta),
        }


def bmeasure_suite(
    ref_bc: BC,
    est_bc: BC,
    track_id: str = "tid",
    windows: tuple = (0.5, 1.5, 3),
):
    records = []
    for matched_poa in [True, False]:
        for window in windows:
            scores = bmeasure(ref_bc, est_bc, window=window, matched_poa=matched_poa)
            for metric, score in scores.items():
                records.append(
                    {
                        "track_id": track_id,
                        "metric": metric.split("_")[0],
                        "prf": metric.split("_")[1],
                        "matched_poa": matched_poa,
                        "window": window,
                        "score": float(score),
                    }
                )

    df = pd.DataFrame(records)
    return (
        df.assign(
            metric=lambda s: s["metric"].mask(
                s["matched_poa"] & s["metric"].isin(["poa", "b"]), s["metric"] + "-m"
            )
        )
        .drop(columns="matched_poa")
        .drop_duplicates()
        .sort_values(by=["track_id", "window", "prf", "metric"])
        .reset_index(drop=True)
    )
