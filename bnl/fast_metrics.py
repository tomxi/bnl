import mir_eval
import numpy as np
from scipy.interpolate import interp1d


def vmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0):
    # adjust timespan of estimations relative to reference
    ref_itvls, ref_labels, est_itvls, est_labels = _align_hier(
        [ref_itvls], [ref_labels], [est_itvls], [est_labels]
    )
    # They are depth 1 hieraries right now, let's get them out of the list
    ref_itvls = ref_itvls[0]
    ref_labels = ref_labels[0]
    est_itvls = est_itvls[0]
    est_labels = est_labels[0]

    precision = 1.0 - conditional_entropy(
        est_itvls, est_labels, ref_itvls, ref_labels
    ) / entropy(est_itvls, est_labels)
    recall = 1.0 - conditional_entropy(
        ref_itvls, ref_labels, est_itvls, est_labels
    ) / entropy(ref_itvls, ref_labels)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def pairwise(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0):
    seg_dur, meet_ref, meet_est = _get_common_grid_meet_matrices(
        [ref_itvls], [ref_labels], [est_itvls], [est_labels]
    )
    meet_both = meet_ref * meet_est

    seg_area = np.outer(seg_dur, seg_dur)
    ref_area = np.sum(meet_ref * seg_area)
    est_area = np.sum(meet_est * seg_area)
    intersetion_area = np.sum(meet_both * seg_area)

    precision = intersetion_area / est_area
    recall = intersetion_area / ref_area
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def lmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0, mono=False):
    # build common grid and meet mats first
    seg_dur, meet_ref, meet_est = _get_common_grid_meet_matrices(
        ref_itvls, ref_labels, est_itvls, est_labels, mono=mono
    )
    seg_area = np.outer(seg_dur, seg_dur)

    recall = triplet_recall(meet_ref, meet_est, seg_area, seg_dur)
    precision = triplet_recall(meet_est, meet_ref, seg_area, seg_dur)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def _meet(gridded_hier_labels, compare_func=np.equal, mono=False):
    """
    Compute the meet matrix for a hierarchy of labels.
    hier_labels.shape = (depth, n_seg). output shape = (n_seg, n_seg)
    compare_func needs to support numpy broadcasting.
    """
    hier_labels = np.array(gridded_hier_labels)
    # Using broadcasting to compute the outer comparison for each level.
    meet_per_level = compare_func(hier_labels[:, :, None], hier_labels[:, None, :])
    max_depth = meet_per_level.shape[0]

    # Create an array representing the level numbers (starting from 1)
    level_indices = np.arange(1, max_depth + 1)[:, None, None]
    if not mono:
        # Deepest level where the labels meet
        depths = np.max(meet_per_level * level_indices, axis=0)
    else:
        # Shallowest level where the labels stops meeting
        depths = max_depth - np.max(
            ~meet_per_level * np.flip(level_indices, axis=0), axis=0
        )

    return depths.astype(int)


def _get_common_grid_meet_matrices(
    ref_itvls, ref_labels, est_itvls, est_labels, mono=False
):
    # Strategy: cut up into common boundaries
    common_itvls, ref_labs, est_labs = _common_grid_itvls_labels(
        *_align_hier(ref_itvls, ref_labels, est_itvls, est_labels)
    )
    # get the meet matrix
    meet_ref = _meet(ref_labs, mono=mono)
    meet_est = _meet(est_labs, mono=mono)
    common_seg_dur = np.diff(common_itvls, axis=1).flatten()
    return common_seg_dur, meet_ref, meet_est


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


def _common_boundaries(list_of_itvls):
    # Get the boundaries of both sets of intervals
    bs = set()
    for itvls in list_of_itvls:
        bs = bs.union(set(mir_eval.util.intervals_to_boundaries(itvls)))
    return np.array(sorted(bs))


def triplet_recall(meet_ref, meet_est, seg_area, seg_dur):
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


def entropy(seg_dur, labels):
    """
    Compute the entropy of a segmentation
    """
    # We get label and duration for each segment, aggrecate duration for each label,
    # and compute entropy by assuming a uniform distribution over the union of all intervals provided.
    # accumulate duration for each unique label
    seg_dur = np.array(seg_dur)
    labels = np.array(labels)
    # check if the user provided mir_eval style flat itvls instead, in that case, get the duration
    if seg_dur.ndim == 2 and seg_dur.shape[1] == 2:
        seg_dur = np.diff(seg_dur, axis=1).flatten()
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    label_durs = np.zeros(len(unique_labels))
    # iterate through segment and find the right label to add duration
    for i, duration in enumerate(seg_dur):
        label_durs[inverse_indices[i]] += duration

    # get ride of zero duration labels
    pi = label_durs[label_durs > 0]
    pi_sum = seg_dur.sum()
    if pi_sum == 0:
        return np.nan
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))


def joint_entropy(itvls1, labels1, itvls2, labels2):
    common_itvls, lab1, lab2 = _common_grid_itvls_labels(
        [itvls1], [labels1], [itvls2], [labels2]
    )

    lab1 = np.array(lab1).flatten()
    lab2 = np.array(lab2).flatten()
    # using numpy broadcasting to get the label pairs
    concurrent_label_pairs = np.char.add(lab1, lab2)

    # get the segment durations and areas These are the joint probabilities of the labels
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    return entropy(seg_dur, concurrent_label_pairs)


def conditional_entropy(itvls, labels, cond_itvls, cond_labels):
    """
    H(S | S_cond) = H(S, S_cond) - H(S_cond)
    """
    joint_ent = joint_entropy(itvls, labels, cond_itvls, cond_labels)
    cond_ent = entropy(np.diff(cond_itvls, axis=1).flatten(), cond_labels)
    return joint_ent - cond_ent


def labels_at_ts(hier_itvls: list, hier_labels: list, ts: np.ndarray):
    """
    get label at ts for all levels in a hierarchy
    """
    results = []
    for itvls, labs in zip(hier_itvls, hier_labels):
        result = _label_at_ts(itvls, labs, ts)
        results.append(result)
    return results


def _label_at_ts(itvls, labels, ts):
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


def _align_hier(ref_itvls, ref_labels, est_itvls, est_labels):
    # First, find the maximum length of the reference
    _, t_end = mir_eval.hierarchy._hierarchy_bounds(ref_itvls)

    # Pre-process the intervals to match the range of the reference,
    # and start at 0
    new_ref_itvls, new_ref_labels = mir_eval.hierarchy._align_intervals(
        ref_itvls, ref_labels, t_min=0.0, t_max=None
    )
    new_est_itvls, new_est_labels = mir_eval.hierarchy._align_intervals(
        est_itvls, est_labels, t_min=0.0, t_max=t_end
    )
    return new_ref_itvls, new_ref_labels, new_est_itvls, new_est_labels
