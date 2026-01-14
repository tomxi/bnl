import frameless_eval as fle
import mir_eval
import numpy as np
import pandas as pd
from mir_eval.util import f_measure, match_events

from .core import BoundaryContour as BC
from .core import MultiSegment as MS


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


def bmeasure2(
    ref_bc: BC,
    est_bc: BC,
    window: float = 0.5,
    trim: bool = False,
    verbose: bool = False,
    beta: float = 1,
    weighted: bool = True,
):
    """
    Instead of doing the HR and POA separately, let's do HR and POA together.
    """
    # Get matching boundaries
    ref_bs = ref_bc.bs
    est_bs = est_bc.bs

    if trim:
        ref_bs = ref_bs[1:-1]
        est_bs = est_bs[1:-1]
    ref_b_times = [b.time for b in ref_bs]
    est_b_times = [b.time for b in est_bs]
    match_idx = mir_eval.util.match_events(ref_b_times, est_b_times, window)

    # For recall, the denominator is the number of reference boundaries
    # The numerator is the sum of the per boundary POA score in estimate
    rec_denom, prec_denom = len(ref_b_times), len(est_b_times)
    rec_num, prec_num = 0, 0

    # Construct the boundary prominence according to ref_bs for recall
    rec_ref_prom, rec_est_prom = [], []
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
    prec_ref_prom, prec_est_prom = [], []
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
    if verbose:
        print("recall, ref, est", rec_ref_prom, rec_est_prom)
        print("precision, ref, est", prec_ref_prom, prec_est_prom)
    for ref_b_idx, est_b_idx in match_idx:
        ref_b = ref_bs[ref_b_idx]
        est_b = est_bs[est_b_idx]
        # So I want to ask for this hit boundary, how is the ordinal information related to the
        # reference boundary recalled?
        # For all the boundaries in the ref_bs, which ones have salience higher?
        # Do these relationships preserve in the est_bs?
        rec_ref_more_prom = rec_ref_prom > ref_b.salience
        rec_est_more_prom = rec_est_prom > est_b.salience
        rec_ref_less_prom = rec_ref_prom < ref_b.salience
        rec_est_less_prom = rec_est_prom < est_b.salience

        pairs_inverted = (
            rec_ref_more_prom * rec_est_less_prom + rec_ref_less_prom * rec_est_more_prom
        )
        pairs_to_recall = len(ref_bs) - 1
        if verbose:
            print(pairs_inverted.sum(), pairs_to_recall)
        rec_num += safe_div(pairs_to_recall - pairs_inverted.sum(), pairs_to_recall, 1)

        # Do the same on est_b for precision
        prec_ref_more_prom = prec_ref_prom > ref_b.salience
        prec_est_more_prom = prec_est_prom > est_b.salience
        prec_ref_less_prom = prec_ref_prom < ref_b.salience
        prec_est_less_prom = prec_est_prom < est_b.salience

        pairs_inverted = (
            prec_ref_more_prom * prec_est_less_prom + prec_ref_less_prom * prec_est_more_prom
        )
        pairs_predicted = len(est_bs) - 1
        if verbose:
            print(pairs_inverted.sum(), pairs_predicted)
        prec_num += safe_div(pairs_predicted - pairs_inverted.sum(), pairs_predicted, 1)

    if verbose:
        print("recall", rec_num, rec_denom)
        print("precision", prec_num, prec_denom)
    recall = safe_div(rec_num, rec_denom, 1)
    precision = safe_div(prec_num, prec_denom, 1)
    return precision, recall, f_measure(precision, recall, beta=beta)


# region: newest bmeasure


def _build_salience_arrays(ref_bs, est_bs, match_idx):
    """
    Build salience arrays at ref_bs's positions, with matched est saliences (0 for unmatched)

    Args:
        ref_bs: Reference boundaries
        est_bs: Estimate boundaries
        match_idx: ref_idx -> est_idx mapping dictionary

    Returns:
        tuple: (ref_sal, est_sal)
    """

    # Recall arrays: iterate over reference boundaries
    ref_sals = np.array([b.salience for b in ref_bs])
    est_sals = np.array(
        [est_bs[match_idx[i]].salience if i in match_idx else 0.0 for i in range(len(ref_bs))]
    )

    return ref_sals, est_sals


def _pair_inversion_ratio(ref_saliences, est_saliences):
    """
    For ref_saliences and est_saliences of the same length, compare ordinal relationships
    at all positions at once.

    For each position, counts how many pairwise ordinal relationships are inverted
    compared to all other positions.

    Requires ref_saliences be all non-zeros. est_saliences can have zeros.

    Args:
        ref_saliences: Array of reference saliences (non-zeros)
        est_saliences: Array of estimate saliences

    Returns:
        np.ndarray: Number of inverted pairs for each position
    """
    if len(ref_saliences) != len(est_saliences):
        raise ValueError("ref_saliences and est_saliences must have the same length")

    if len(ref_saliences) == 1:
        # no pairs to compare, perfect ordial agreement
        return np.array([0.0])
    else:
        # Create comparison matrices using broadcasting
        # Shape: (n, n) where [i, j] indicates whether position i > position j
        ref_more = ref_saliences[:, np.newaxis] > ref_saliences[np.newaxis, :]
        ref_less = ref_saliences[:, np.newaxis] < ref_saliences[np.newaxis, :]

        est_more_or_eq = est_saliences[:, np.newaxis] >= est_saliences[np.newaxis, :]
        est_less_or_eq = est_saliences[:, np.newaxis] <= est_saliences[np.newaxis, :]

        # Count inversions for each position (row)
        # ref says more but est says less, or vice versa
        inversions = (ref_more & est_less_or_eq) | (ref_less & est_more_or_eq)

        # Sum across columns to get inversion count for each position
        return inversions.sum(axis=1) / (len(ref_saliences) - 1)


def bmeasure3(
    ref,
    est,
    window: float = 0.5,
    trim: bool = True,
    weighted: bool = True,
    verbose: bool = False,
    beta: float = 1.0,
):
    """
    Boundary measure combining hit rate and pairwise ordinal agreement.

    For each matched boundary, computes how well the ordinal relationships
    with all other boundaries are preserved between reference and estimate.

    Args:
        ref: Reference MultiSegment
        est: Estimate MultiSegment
        window: Matching window in seconds
        trim: If True, exclude first and last boundaries
        verbose: If True, print debug information
        beta: F-measure beta parameter

    Returns:
        tuple: (precision, recall, f_measure)
    """
    # check boundary monotonicity
    if not (ref.has_monotonic_bs() and est.has_monotonic_bs()):
        raise ValueError("Both reference and estimate must have monotonic boundaries")

    # turn MS to BCs
    ref_bc = ref.contour("prob")
    est_bc = est.contour("prob")

    # Extract boundaries
    ref_bs = ref_bc.bs
    est_bs = est_bc.bs

    # Optional trimming
    if trim:
        ref_bs = ref_bs[1:-1]
        est_bs = est_bs[1:-1]

    # Match boundaries
    ref_times = [b.time for b in ref_bs]
    est_times = [b.time for b in est_bs]
    match_idx = match_events(ref_times, est_times, window)

    # Build salience arrays for recall
    ref_match_map = {pair[0]: pair[1] for pair in match_idx}
    rec_ref_sals, rec_est_sals = _build_salience_arrays(ref_bs, est_bs, ref_match_map)

    # Build salience arrays for precision
    est_match_map = {pair[1]: pair[0] for pair in match_idx}
    prec_est_sals, prec_ref_sals = _build_salience_arrays(est_bs, ref_bs, est_match_map)

    if verbose:
        print(f"Recall - Ref saliences: {rec_ref_sals}")
        print(f"Recall - Est saliences: {rec_est_sals}")
        print(f"Precision - Ref saliences: {prec_ref_sals}")
        print(f"Precision - Est saliences: {prec_est_sals}")

    # Compute per boundary recall and precision scores
    rec_scores = np.zeros(len(ref_bs))
    prec_scores = np.zeros(len(est_bs))

    # Compute all inversions at once for recall and precision
    rec_ord_scores = 1 - _pair_inversion_ratio(rec_ref_sals, rec_est_sals)
    prec_ord_scores = 1 - _pair_inversion_ratio(prec_est_sals, prec_ref_sals)

    # aggregate matched boundaries' ordinal scores for final recall and precision
    for ref_idx, est_idx in match_idx:
        rec_scores[ref_idx] = rec_ord_scores[ref_idx]
        prec_scores[est_idx] = prec_ord_scores[est_idx]

    if weighted:
        rec_weights = rec_ref_sals / rec_ref_sals.sum()
        prec_weights = prec_est_sals / prec_est_sals.sum()
    else:
        rec_weights = np.ones(len(ref_bs)) / len(ref_bs)
        prec_weights = np.ones(len(est_bs)) / len(est_bs)

    if verbose:
        print("Precision scores:", prec_scores)
        print("Precision weights:", prec_weights)
        print("Recall scores:", rec_scores)
        print("Recall weights:", rec_weights)

    rec = rec_scores @ rec_weights
    prec = prec_scores @ prec_weights
    return prec, rec, f_measure(prec, rec, beta)


# endregion

# region: wrapper around frameless_eval metrics


def lmeasure(ref_ms: MS, est_ms: MS, **kwargs):
    return fle.lmeasure(ref_ms.itvls, ref_ms.labels, est_ms.itvls, est_ms.labels, **kwargs)


def tmeasure(ref_ms: MS, est_ms: MS, **kwargs):
    return mir_eval.hierarchy.tmeasure(ref_ms.itvls, est_ms.itvls, **kwargs)
