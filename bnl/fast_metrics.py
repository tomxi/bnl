import mir_eval
import numpy as np
from scipy.interpolate import interp1d


def _common_boundaries(list_of_itvls):
    # Get the boundaries of both sets of intervals
    bs = set()
    for itvls in list_of_itvls:
        bs = bs.union(set(mir_eval.util.intervals_to_boundaries(itvls)))
    return np.array(sorted(bs))


def _encode_labels(labels):
    """
    Convert a list of string labels to integer labels with a reverse mapping.

    Parameters:
        labels (list/array of str): The input list of string labels.

    Returns:
        tuple: A tuple containing:
            - int_labels (list of int): The list of integer labels.
            - int_to_label (list): List of original labels indexed by their integer codes.
    """
    unique_labels = []
    int_labels = []
    for label in labels:
        label = str(label)
        if label not in unique_labels:
            unique_labels.append(label)
        int_labels.append(unique_labels.index(label))
    return np.array(int_labels, dtype=int), unique_labels


def label_at_ts(itvls: np.ndarray, labels: np.ndarray, ts: np.ndarray, decode=True):
    """
    Label intervals at a specific timestamp.
    Let's us interpolate object
    """
    # We need to convert list of labels to list of integers

    bs = mir_eval.util.intervals_to_boundaries(itvls)
    ts = np.atleast_1d(ts).flatten()
    lab_idx, lab_map = _encode_labels(labels)
    # repeat the last label for the last boundary.
    lab_idx = np.concatenate((lab_idx, [lab_idx[-1]]))

    # Create interpolator that returns None outside the range
    interpolator = interp1d(
        bs, lab_idx, kind="previous", bounds_error=True, assume_sorted=True, copy=False
    )

    # Apply interpolation
    if not decode:
        return interpolator(ts).astype(int)
    else:
        return np.array([lab_map[idx] for idx in interpolator(ts).astype(int)])


def labels_at_ts(hier_itvls: list, hier_labels: list, ts: np.ndarray):
    """
    get label at ts for all levels in a hierarchy
    """
    results = []
    for itvls, labs in zip(hier_itvls, hier_labels):
        result = label_at_ts(itvls, labs, ts)
        results.append(result)
    return results


def _make_common_itvls_grid(
    hier_itvls1,
    hier_labels1,
    hier_itvls2,
    hier_labels2,
    # label_condition,
):
    """Label condition is a element of labels2,
    we slice the itvls and labels to get a new sub segmentation
    based on the conditioned on labels2 = label_condition.
    """
    # Strategy: build a new set of common_intervals, and labels of equal length, then do array indexing.
    # Merge boundaries and compute segment durations.
    common_bs = _common_boundaries(hier_itvls1 + hier_itvls2)
    common_itvls = mir_eval.util.boundaries_to_intervals(common_bs)

    # Find the label at each common boundary.
    gridded_labels1 = labels_at_ts(hier_itvls1, hier_labels1, common_bs[:-1])
    gridded_labels2 = labels_at_ts(hier_itvls2, hier_labels2, common_bs[:-1])
    return common_itvls, gridded_labels1, gridded_labels2


def _meet(gridded_hier_labels, compare_func=np.equal, mono=False):
    # Convert input to a NumPy array.
    hier_labels = np.array(gridded_hier_labels)
    # CHATGPT suggested this
    # Using broadcasting to compute the outer comparison for each level.
    # hier_labels has shape (depth, n_seg) and the operation below yields
    # an array of shape (depth, n_seg, n_seg) with the pairwise comparisons.
    meet_per_level = compare_func(hier_labels[:, :, None], hier_labels[:, None, :])
    depth = meet_per_level.shape[0]
    # Create an array representing the level numbers (starting from 1)
    level_indices = np.arange(1, depth + 1)[:, np.newaxis, np.newaxis]

    if not mono:
        # Where meet_per_level is False, multiply by 0; where True, multiply by level number
        # Then take the maximum along the depth axis
        depths = np.max(meet_per_level * level_indices, axis=0)
    else:
        # find first level that's False
        depths = depth - np.max(
            ~meet_per_level * np.flip(level_indices, axis=0), axis=0
        )

    return depths.astype(int)


def lmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0, mono=False):
    # build common grid and meet mats first
    common_itvls, ref_labels, est_labels = _make_common_itvls_grid(
        ref_itvls, ref_labels, est_itvls, est_labels
    )

    # make meet matrices once
    meet_ref = _meet(ref_labels, mono=mono)
    meet_est = _meet(est_labels, mono=mono)
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    seg_area = np.outer(seg_dur, seg_dur)

    recall = _triplet_recall(meet_ref, meet_est, seg_area, seg_dur)
    precision = _triplet_recall(meet_est, meet_ref, seg_area, seg_dur)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def _triplet_recall(meet_ref, meet_est, seg_area, seg_dur):
    per_segment_recall = []
    for seg_idx in range(len(seg_dur)):
        # get per segment recall
        per_segment_recall.append(
            _segment_triplet_recall(meet_ref, meet_est, seg_idx, seg_area)
        )

    per_segment_recall = np.array(per_segment_recall)
    # normalize by duration.
    valid_seg = ~np.isnan(per_segment_recall)
    if np.sum(seg_dur[valid_seg]) == 0:
        return 0.0
    else:
        return np.sum(per_segment_recall[valid_seg] * seg_dur[valid_seg]) / np.sum(
            seg_dur[valid_seg]
        )


def _segment_triplet_recall(meet_ref, meet_est, seg_idx, seg_area):
    # given a segment idx, their relevance against each other segment.
    ref_rel_againt_seg_i = meet_ref[seg_idx, :]
    est_rel_againt_seg_i = meet_est[seg_idx, :]

    pairs_to_recall = _meet([ref_rel_againt_seg_i], compare_func=np.greater)
    pairs_recalled = (
        _meet([est_rel_againt_seg_i], compare_func=np.greater) * pairs_to_recall
    )

    area_to_recall = np.sum(pairs_to_recall * seg_area)
    area_recalled = np.sum(pairs_recalled * seg_area)

    return area_recalled / area_to_recall if area_to_recall > 0 else np.nan
