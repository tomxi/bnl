# So I want continuous version of L measure and T measure.

# Let's start with the continuous version of the relevance curve given q and the structure matrix it induces:

# To represent a contions relevance curve, we use the S object

from . import S, H, utils, fio
import xarray as xr
import os
from mir_eval.util import boundaries_to_intervals, f_measure
from mir_eval import hierarchy as meh
import numpy as np
import time

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")


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
    # We return a segmentation with all the boundaries from the hierarchy
    # and store the relevance value as the label.
    return S(boundaries_to_intervals(hier.beta), relevance)


def recall_at_t(
    h_ref: H,
    h_est: H,
    t: float,
    meet_mode: str = "deepest",
    window: float = 0,
    transitive: bool = True,
    debug=0,  # 0 for no debug, 1 for debug, 2 for debug with mats
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
    # get beta (set of boundaries) from both s_ref and s_est.
    # Between these boundaries things are piecewise constant.
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

    # S.A returns all segment pair comparison with compare_fn.
    # When compare_fn is np.equal, S.A is the meet matrix: SSM between two list of segment labels.
    # We use np.greater here to get the orientation of the triplets
    positions_to_recall = s_ref.A(bs=common_bs, compare_fn=compare_fn)
    # Anything greater is recalled, even when transitive
    positions_recalled = (
        s_est.A(bs=common_bs, compare_fn=np.greater) * positions_to_recall
    )

    # Calculate the area of the grid made by segment boundaries
    common_grid_area = utils.bs2grid_area(common_bs)

    if debug == 2:
        return dict(
            iota=positions_to_recall,
            alpha=positions_recalled,
            bs=common_bs,
            grid_area=common_grid_area,
        )
    area_to_recall = np.sum(positions_to_recall * common_grid_area)
    area_recalled = np.sum(positions_recalled * common_grid_area)
    max_area = np.sum(common_grid_area) / 2.0
    if debug == 1:
        return area_recalled / max_area, area_to_recall / max_area
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


def lmeasure(h_ref: H, h_est: H, meet_mode: str = "deepest", beta=1.0, **kwargs):
    p = precision(h_ref, h_est, meet_mode=meet_mode, **kwargs)
    r = recall(h_ref, h_est, meet_mode=meet_mode, **kwargs)
    f = f_measure(p, r, beta=beta)
    return (p, r, f)


def align_hier(h_ref: H, h_est: H):
    # First, find the maximum length of the reference
    _, t_end = meh._hierarchy_bounds(h_ref.itvls)

    # Pre-process the intervals to match the range of the reference,
    # and start at 0
    new_h_ref = H(
        *meh._align_intervals(h_ref.itvls, h_ref.labels, t_min=0.0, t_max=None)
    )
    new_h_est = H(
        *meh._align_intervals(h_est.itvls, h_est.labels, t_min=0.0, t_max=t_end)
    )
    return new_h_ref, new_h_est


def evaluate(h_ref, h_est, **kwargs):
    """
    Evaluate the precision, recall, and F-measure between two hierarchies.
    """
    h_ref, h_est = align_hier(h_ref, h_est)
    return lmeasure(h_ref, h_est, **kwargs)


def time_lmeasure(ref, est, frame_size=0):
    """
    Measure the time taken to compute the lmeasure metric using both implementations.

    Parameters:
    - ref: Reference hierarchy
    - est: Estimated hierarchy
    - frame_size: frame_size for the metric calculation. 0 will use my implementation

    Returns:
    - Time taken for each implementation
    """
    # Pad and align the hierarchies
    ref, est = align_hier(ref, est)
    # Measure time for different implementation with frame size or not
    start_time = time.time()
    if frame_size == 0:
        results = lmeasure(ref, est)
    else:
        results = meh.lmeasure(
            ref.itvls, ref.labels, est.itvls, est.labels, frame_size=frame_size
        )
    run_time = time.time() - start_time
    return run_time, results


def time_salami_track(tid):
    hiers = fio.salami_ref_hiers(tid=tid)
    if len(hiers) < 2:
        return

    fname = f"./compare_implementation/{tid}.nc"
    if os.path.exists(fname):
        return
    test_frame_size = [0, 0.1, 0.25, 0.5, 1, 2]
    da_coords = dict(frame_size=test_frame_size, output=["run_time", "lp", "lr", "lm"])
    # Create a dataarray for this track's results
    result_da = xr.DataArray(dims=da_coords.keys(), coords=da_coords)

    # Get the two hierarchies
    ref = hiers[0]
    est = hiers[1]

    for frame_size in test_frame_size:
        # Measure time for both implementations
        run_time, results = time_lmeasure(ref, est, frame_size=frame_size)
        result_da.loc[dict(frame_size=frame_size)] = [run_time, *results]

    # save the results
    result_da.to_netcdf(fname)


# Modified from mir_eval.hierarchy
def gauc_t(meet_mat_ref, meet_mat_est, transitive=True, window=None):
    """
    Compute ranking recall and normalizer for each query position.

    Parameters:
    -----------
    meet_mat_ref : scipy.sparse matrix
        Reference meet matrix
    meet_mat_est : scipy.sparse matrix
        Estimated meet matrix
    transitive : bool
        If True, then transitive comparisons are counted, meaning that
        ``(q, i)`` and ``(q, j)`` can differ by any number of levels.
        If False, then ``(q, i)`` and ``(q, j)`` can differ by exactly one
        level.
    window : number or None
        The maximum number of frames to consider for each query.
        If `None`, then all frames are considered.

    Returns:
    --------
    q_ranking_inversions : numpy.ndarray
        Ranking inversions for each query position
    q_ranking_normalizer : numpy.ndarray
        Normalizer for each query position
    """
    # Make sure we have the right number of frames
    if meet_mat_ref.shape != meet_mat_est.shape:
        raise ValueError(
            "Estimated and reference hierarchies " "must have the same shape."
        )

    # How many frames?
    n = meet_mat_ref.shape[0]

    # By default, the window covers the entire track
    if window is None:
        window = n

    q_ranking_normalizer = np.zeros(n)
    q_ranking_inversions = np.zeros(n)
    q_ranking_allpairs = np.zeros(n)

    for query in range(n):
        # Get the window around the query
        win_slice = slice(max(0, query - window), min(n, query + window))
        ref_window = meet_mat_ref[query, win_slice].toarray().ravel()
        est_window = meet_mat_est[query, win_slice].toarray().ravel()
        # get the query'th row
        q_window_ref = np.delete(ref_window, query)
        q_window_est = np.delete(est_window, query)
        # count ranking violations
        inversions, normalizer = meh._compare_frame_rankings(
            q_window_ref, q_window_est, transitive=transitive
        )
        q_ranking_inversions[query] = inversions
        q_ranking_normalizer[query] = normalizer
        # n choice 2 is the total number of pairs
        q_ranking_allpairs[query] = len(q_window_ref) * (len(q_window_est) - 1) / 2

    return q_ranking_inversions, q_ranking_normalizer, q_ranking_allpairs
