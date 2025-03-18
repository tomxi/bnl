# So I want continuous version of L measure and T measure.

# Let's start with the continuous version of the relevance curve given q and the structure matrix it induces:

# To represent a contions relevance curve, we use the S object

from . import S, H, utils, fio
import xarray as xr
import os
from mir_eval.util import boundaries_to_intervals, f_measure
from mir_eval import hierarchy
import numpy as np
import time


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
    positions_recalled = (
        s_est.A(bs=common_bs, compare_fn=np.greater) * positions_to_recall
    )
    # Calculate the area of the grid made by segment boundaries
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


def lmeasure(h_ref: H, h_est: H, meet_mode: str = "deepest", beta=1.0):
    p = precision(h_ref, h_est, meet_mode=meet_mode)
    r = recall(h_ref, h_est, meet_mode=meet_mode)
    f = f_measure(p, r, beta=beta)
    return (p, r, f)


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
    # Measure time for different implementation with frame size or not
    start_time = time.time()
    if frame_size == 0:
        results = lmeasure(ref, est)
    else:
        results = hierarchy.lmeasure(
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
