import mir_eval
import numpy as np
from scipy.interpolate import interp1d


def labels_at_ts(hier_itvls: list, hier_labels: list, ts: np.ndarray):
    """
    get label at ts for all levels in a hierarchy
    """
    results = []
    for itvls, labs in zip(hier_itvls, hier_labels):
        result = _label_at_ts(itvls, labs, ts)
        results.append(result)
    return results


def _common_boundaries(list_of_itvls):
    # Get the boundaries of both sets of intervals
    bs = set()
    for itvls in list_of_itvls:
        bs = bs.union(set(mir_eval.util.intervals_to_boundaries(itvls)))
    return np.array(sorted(bs))


def _common_grid_itvls_labels(
    hier_itvls1,
    hier_labels1,
    hier_itvls2,
    hier_labels2,
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
    """
    Compute the meet matrix for a hierarchy of labels.
    hier_labels.shape = (depth, n_seg).
    """
    # Convert input to a NumPy array.
    hier_labels = np.array(gridded_hier_labels)
    # CHATGPT suggested this
    # Using broadcasting to compute the outer comparison for each level.
    # hier_labels has shape (depth, n_seg) and the operation below yields
    # an array of shape (depth, n_seg, n_seg) with the pairwise comparisons.
    meet_per_level = compare_func(hier_labels[:, :, None], hier_labels[:, None, :])
    max_depth = meet_per_level.shape[0]

    # Create an array representing the level numbers (starting from 1)
    level_indices = np.arange(1, max_depth + 1)[:, None, None]
    if not mono:
        # Where meet_per_level is False, multiply by 0; where True, multiply by level number
        # Then take the maximum along the depth axis
        depths = np.max(meet_per_level * level_indices, axis=0)
    else:
        # find first level that's False.
        # look from the bottom up and use the same logic as above, changing True to False, shallow to deep.
        depths = max_depth - np.max(
            ~meet_per_level * np.flip(level_indices, axis=0), axis=0
        )

    return depths.astype(int)


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


def lmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0, mono=False):
    # build common grid and meet mats first
    common_itvls, ref_labels, est_labels = _common_grid_itvls_labels(
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


def pairwise(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0):
    # Strategy: cut up into common boundaries
    common_itvls, ref_labs, est_labs = _common_grid_itvls_labels(
        [ref_itvls], [ref_labels], [est_itvls], [est_labels]
    )
    # get the segment durations and areas
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    seg_area = np.outer(seg_dur, seg_dur)
    # get the meet matrix
    meet_ref = _meet(ref_labs)
    meet_est = _meet(est_labs)
    meet_both = meet_ref * meet_est

    ref_area = np.sum(meet_ref * seg_area)
    est_area = np.sum(meet_est * seg_area)
    intersetion_area = np.sum(meet_both * seg_area)
    precision = intersetion_area / est_area if est_area > 0 else np.nan
    recall = intersetion_area / ref_area if ref_area > 0 else np.nan
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def vmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0):

    return None


def entropy(itvls, labels):
    """
    Compute the entropy of a segmentation
    """
    # We get label and duration for each segment, aggrecate duration for each label,
    # and compute entropy by assuming a uniform distribution over the union of all intervals provided.
    seg_dur = np.diff(itvls, axis=1).flatten()
    pi_sum = seg_dur.sum()
    if pi_sum == 0.0:
        return 1.0

    # accumulate duration for each unique label
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    label_durs = np.zeros(len(unique_labels))
    # iterate through segment and find the right label to add duration
    for i, duration in enumerate(seg_dur):
        label_durs[inverse_indices[i]] += duration

    # get ride of zero duration labels
    pi = label_durs[label_durs > 0]
    pi_sum = seg_dur.sum()
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))


def joint_entropy(itvls1, labels1, itvls2, labels2):
    common_itvls, lab1, lab2 = _common_grid_itvls_labels(
        [itvls1], [labels1], [itvls2], [labels2]
    )
    # get the segment durations and areas
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    # THese are the joint probabilities of the labels
    joint_prob = np.outer(seg_dur, seg_dur) / np.sum(seg_dur) ** 2
    # These are the label pairs themselves
    label_pairs = np.add.outer(lab1, lab2, casting="unsafe", dtype=str)
    return joint_prob, label_pairs


def label_at_ts(itvls, labels, ts):
    """
    Label intervals at specific timestamps.

    This helper converts string labels to integer codes, interpolates these codes
    over interval boundaries, and optionally decodes the integer labels back to the original strings.

    Parameters:
        itvls (np.ndarray): An array of intervals (shape: [n, 2]).
        labels (list/array of str): The labels for each interval.
        ts (np.ndarray): Timestamps at which to evaluate the labels.
        decode (bool, optional): If True, returns the original labels; otherwise, returns integer codes.
                                   Defaults to True.

    Returns:
        np.ndarray: An array of labels (either original or integer-coded) corresponding to each timestamp in ts.
    """

    # Helper: encode string labels to integers with a reverse mapping.
    def _encode_labels(lbls):
        unique = []
        codes = []
        for lbl in lbls:
            lbl = str(lbl)
            if lbl not in unique:
                unique.append(lbl)
            codes.append(unique.index(lbl))
        return np.array(codes, dtype=int), unique

    # Convert intervals to boundaries and flatten timestamps.
    boundaries = mir_eval.util.intervals_to_boundaries(itvls)
    ts = np.atleast_1d(ts).flatten()

    # Encode labels to integer codes.
    int_labels, label_map = _encode_labels(labels)
    # Append the last label to cover the final boundary.
    int_labels = np.concatenate((int_labels, [int_labels[-1]]))

    interp = interp1d(
        boundaries,
        int_labels,
        kind="previous",
        bounds_error=True,
        assume_sorted=True,
        copy=False,
    )

    # Return decoded labels
    return np.array([label_map[i] for i in interp(ts).astype(int)])


def faster_label_at_ts(itvls, labels, ts):
    """
    Assign labels to timestamps using interval boundaries and vectorized lookup.

    Parameters
    ----------
    itvls : np.ndarray
        An array of shape (n, 2) representing intervals.
    labels : list
        A list of labels corresponding to each interval.
    ts : array-like
        Timestamps to be labeled.

    Returns
    -------
    np.ndarray
        An array of labels corresponding to each timestamp in ts.
    """
    boundaries = mir_eval.util.intervals_to_boundaries(itvls)
    extended = np.array(labels + [labels[-1]])  # repeat last label for last boundary
    return extended[np.searchsorted(boundaries, np.atleast_1d(ts), side="right") - 1]
