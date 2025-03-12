# So I want continuous version of L measure and T measure.

# Let's start with the continuous version of the relevance curve given q and the structure matrix it induces:

# To represent a contions relevance curve, we use the S object

from . import S, H, utils
from mir_eval.util import boundaries_to_intervals
import numpy as np


def get_segment_relevence(hier: H, t: float, meet_mode="deepest"):
    """
    Get the relevance curve for a given query time point t in seconds.

    Parameters
    ----------
    hier : Hierarchy
        The hierarchical structure.
    t : float
        The query time point.
    meet_mode : str
        The meeting mode to use for relevance calculation.
        Options are "deepest", "mono", or "mean".
    Returns
    -------
    S
        The relevance curve as an S segment object.
    """
    ts = (hier.beta[:-1] + hier.beta[1:]) / 2.0
    relevance = [hier.meet(t, m, mode=meet_mode) for m in ts]
    return S(boundaries_to_intervals(hier.beta), relevance)


def recall_at_t(
    h_ref: H,
    h_est: H,
    t: float,
    meet_mode: str = "deepest",
    window: float = 15,
    transitive: bool = False,
):
    """
    Compute recall at time t for a reference and estimated hierarchy.

    Parameters
    ----------
    h_ref : Hierarchy
        The reference hierarchy.
    h_est : Hierarchy
        The estimated hierarchy.
    t : float
        The time at which to compute recall.

    Returns
    -------
    float
        The recall at time t.
    """
    s_ref = get_segment_relevence(h_ref, t, meet_mode=meet_mode)
    s_est = get_segment_relevence(h_est, t, meet_mode=meet_mode)
    common_bs = sorted(list(set(s_ref.beta).union(s_est.beta)))
    min_t = min(common_bs)
    max_t = max(common_bs)

    if window:
        # add t +- window to common_bs and get rid of all bs outside of this range
        common_bs = [t - window] + common_bs + [t + window]
        common_bs = [
            b
            for b in common_bs
            if (b >= t - window) and (b <= t + window) and (b >= min_t) and (b <= max_t)
        ]
    if transitive:
        compare_fn = np.greater
    else:
        # They have to be greater by exactly 1
        compare_fn = np.frompyfunc(lambda x, y: int(x) - int(y) == 1, 2, 1)

    positions_to_recall = s_ref.A(bs=common_bs, compare_fn=compare_fn)
    positions_recalled = (
        s_est.A(bs=common_bs, compare_fn=np.greater) * positions_to_recall
    )
    common_grid_area = utils.bs2grid_area(common_bs)
    area_to_recall = np.sum(positions_to_recall * common_grid_area)
    area_recalled = np.sum(positions_recalled * common_grid_area)
    return area_recalled / area_to_recall if area_to_recall > 0 else np.nan


def recall(
    h_ref: H,
    h_est: H,
    meet_mode: str = "deepest",
    window: float = 0,
    transitive: bool = True,
):
    common_bs = np.array(sorted(list(set(h_ref.beta).union(h_est.beta))))
    common_ts = (common_bs[:-1] + common_bs[1:]) / 2.0

    per_segment_recall = np.array(
        [
            recall_at_t(
                h_ref,
                h_est,
                t,
                meet_mode=meet_mode,
                window=window,
                transitive=transitive,
            )
            for t in common_ts
        ]
    )
    seg_dur = np.diff(common_bs)

    # if per_segment_recall is nan, ignore segment in calculation
    valid_seg = ~np.isnan(per_segment_recall)
    if np.sum(seg_dur[valid_seg]) == 0:
        return 0.0
    else:
        return np.sum(per_segment_recall[valid_seg] * seg_dur[valid_seg]) / np.sum(
            seg_dur[valid_seg]
        )


def precision(
    h_ref: H,
    h_est: H,
    meet_mode: str = "deepest",
    window: float = 0,
    transitive: bool = True,
):
    return recall(
        h_est, h_ref, meet_mode=meet_mode, window=window, transitive=transitive
    )
