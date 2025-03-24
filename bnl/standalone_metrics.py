import mir_eval
import numpy as np


def common_boundaries(list_of_itvls):
    # Get the boundaries of both sets of intervals
    bs = set()
    for itvls in list_of_itvls:
        bs = bs.union(set(mir_eval.util.intervals_to_boundaries(itvls)))
    return np.array(sorted(bs))


# Let's combine and optimize these two functions below later
def labels_at_t(hier_itvls, hier_labels, t):
    """
    Get the labels at a specific time t for hierarchical intervals.
    returns a list of labels at time t for each level.
    """
    # we ensure t is in the range of the intervals.
    if len(hier_itvls) == 0:
        return None
    elif t < hier_itvls[0][0][0] or t > hier_itvls[0][-1][-1]:
        return [None] * len(hier_labels)
    elif t == hier_itvls[0][-1][-1]:
        return [labels[-1] for labels in hier_labels]

    seg_idx = []
    # Get the labels at time t for each level using searchsorted
    for itvls, labels in zip(hier_itvls, hier_labels):
        # Get the index of the interval that contains t
        start_times = [itvl[0] for itvl in itvls]
        seg_idx.append(np.searchsorted(start_times, t, side="right") - 1)

    return [labels[idx] for idx, labels in zip(seg_idx, hier_labels)]


def label_at_ts(itvls, labels, ts):
    """Get labels at multiple time points for flat (non-hierarchical) intervals.

    Parameters:
        itvls: List of [start, end] intervals
        labels: List of corresponding labels
        ts: Time point(s) to query

    Returns:
        List of labels at each queried time point, empty list if no interval contains the time
    """
    starts = np.array([iv[0] for iv in itvls])
    ends = np.array([iv[1] for iv in itvls])

    # Use vectorized operations to find labels at each time point
    idx = np.vectorize(
        lambda t: np.searchsorted(starts, t, side="right") - 1,
        otypes=[np.int_],
        signature="()->()",
    )(ts)
    mask = (idx >= 0) & (ts < ends[idx])
    results = [labels[i] if m else None for i, m in zip(idx, mask)]

    return results


def meet(hier_itvls, hier_labels, u, v, mode="deepest", compare_fn=np.equal):
    """
    Compute the meeting point for a list of given time pairs (u, v) and mode.
    use compare_fn on labels at u and v for defining meet.

    Parameters
    ----------
    hier_itvls : list of list of pairs
        The hierarchical intervals.
    hier_labels : list of list of values.
        The labels.
    u : float
        The first time point.
    v : float
        The second time point.
    mode : str
        The meeting mode to use.
        Options are "deepest", "mono"
    Returns
    -------
    np.array
        The relevance calculated at the specified time pairs.
        The degree of relevance, or depth of meet.
    """
    # Strategy: build a num_pairs x num_level matrix, and record the meeting point

    # First get the labels for u and v at each level
    u_labels = np.array(labels_at_t(hier_itvls, hier_labels, u))
    v_labels = np.array(labels_at_t(hier_itvls, hier_labels, v))

    # Then compare them with the compare_fn
    lvl_meet = np.atleast_1d(np.vectorize(compare_fn)(u_labels, v_labels))

    if mode == "deepest":
        # Find the idx of the Last True value or zero if all are False
        return len(lvl_meet) - np.argmax(lvl_meet[::-1]) if lvl_meet.any() else 0
    elif mode == "mono":
        # Find the idx of the first False value, len(lvl_meet) if all are True
        return np.argmax(lvl_meet == False) if not lvl_meet.all() else len(lvl_meet)
    else:
        raise ValueError(f"Unknown meeting mode: {mode}.\n Use 'deepest' or 'mono'.")


def relevance_hierarchy_at_t(hier_itvls, hier_labels, t, bs=None, meet_mode="deepest"):
    """
    Get the relevance curve for a given query time point t in seconds.

    Parameters
    ----------
    itvls : list of list of pairs
        The hierarchical intervals.
    labels : list of list of values.
        The labels.
    t : float
        The query time point.
    meet_mode : str
        The meeting mode to use for relevance calculation.
        Options are "deepest", "mono".
    Returns
    -------
    itvls: list of pairs,
        The intervals and the relevance values.
    relevances: list of values.
    """
    # merge list of boundaries into a single list and sort
    if bs is None:
        bs = common_boundaries(hier_itvls)
    # get the meet at time t, all_bs[:-1], thats the relevance of t against each segment.

    rel_val = np.vectorize(
        lambda u: float(
            meet(hier_itvls, hier_labels, t, u, meet_mode, compare_fn=np.equal)
        ),
        otypes=[np.float64],
        signature="()->()",
    )(bs[:-1])

    return mir_eval.util.boundaries_to_intervals(bs), rel_val


def triplet_recall_at_t(
    ref_hier_itvls,
    ref_hier_labels,
    est_hier_itvls,
    est_hier_labels,
    t,
    meet_mode="deepest",
    transitive=True,
    debug=0,
):
    """
    Compute recall at time t for a reference and estimated hierarchy.

    Parameters
    ----------
    ref_itvls : list
        The reference intervals.
    ref_labels : list
        The labels corresponding to the reference intervals.
    est_itvls : list
        The estimated intervals.
    est_labels : list
        The labels corresponding to the estimated intervals.
    t : float
        The time at which to compute recall.

    Returns
    -------
    float
        The recall at time t.
    """
    common_bs = common_boundaries(ref_hier_itvls + est_hier_itvls)
    min_t = common_bs[0]
    max_t = common_bs[-1]

    common_itvls, ref_rel = relevance_hierarchy_at_t(
        ref_hier_itvls, ref_hier_labels, t, bs=common_bs, meet_mode=meet_mode
    )
    common_itvls, est_rel = relevance_hierarchy_at_t(
        est_hier_itvls, est_hier_labels, t, bs=common_bs, meet_mode=meet_mode
    )

    if transitive:
        sig_compare_fn = np.greater
    else:
        # They have to be greater by exactly 1
        sig_compare_fn = np.frompyfunc(lambda x, y: int(x) - int(y) == 1, 2, 1)

    positions_to_recall = sig_compare_fn.outer(ref_rel, ref_rel).astype(float)
    positions_recalled = (
        np.greater.outer(est_rel, est_rel).astype(float) * positions_to_recall
    )
    seg_dur = np.diff(common_bs)
    common_grid_area = np.outer(seg_dur, seg_dur)

    if debug == 2:
        return dict(
            iota=positions_to_recall,
            alpha=positions_recalled,
            bs=common_bs,
            grid_area=common_grid_area,
        )
    area_to_recall = np.sum(positions_to_recall * common_grid_area)
    area_recalled = np.sum(positions_recalled * common_grid_area)
    if debug == 1:
        max_area = ((max_t - min_t) ** 2) / 2.0
        return area_recalled / max_area, area_to_recall / max_area
    return area_recalled / area_to_recall if area_to_recall > 0 else np.nan


def triplet_recall(
    ref_hier_itvls,
    ref_hier_labels,
    est_hier_itvls,
    est_hier_labels,
    meet_mode="deepest",
    transitive=True,
    debug=0,
):
    common_bs = common_boundaries(ref_hier_itvls + est_hier_itvls)
    per_segment_recall = np.vectorize(
        lambda t: triplet_recall_at_t(
            ref_hier_itvls,
            ref_hier_labels,
            est_hier_itvls,
            est_hier_labels,
            t,
            meet_mode=meet_mode,
            transitive=transitive,
            debug=debug,
        )
    )(common_bs[:-1])
    seg_dur = np.diff(common_bs)

    # if per_segment_recall is nan, ignore segment in calculation
    valid_seg = ~np.isnan(per_segment_recall)
    if np.sum(seg_dur[valid_seg]) == 0:
        return 0.0
    else:
        return np.sum(per_segment_recall[valid_seg] * seg_dur[valid_seg]) / np.sum(
            seg_dur[valid_seg]
        )


def pairwise_recall(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
):
    """
    Compute pairwise recall at time t for a reference and estimated labeled segmentation.

    Parameters
    ----------
    ref_itvls : list
        The reference intervals.
    ref_labels : list
        The labels corresponding to the reference intervals.
    est_itvls : list
        The estimated intervals.
    est_labels : list
        The labels corresponding to the estimated intervals.

    Returns
    -------
    float
        The recall at time t.
    """
    # Find the common boundaries
    common_bs = common_boundaries([ref_itvls, est_itvls])

    # Get the label at each common_bs
    # Get the label for each common boundary using label_at_ts
    ref_labels = np.array(label_at_ts(ref_itvls, ref_labels, common_bs[:-1]))
    est_labels = np.array(label_at_ts(est_itvls, est_labels, common_bs[:-1]))

    positions_to_recall = np.equal.outer(ref_labels, ref_labels).astype(float)
    positions_recalled = (
        np.equal.outer(est_labels, est_labels).astype(float) * positions_to_recall
    )
    seg_dur = np.diff(common_bs)
    common_grid_area = np.outer(seg_dur, seg_dur)
    area_to_recall = np.sum(positions_to_recall * common_grid_area)
    area_recalled = np.sum(positions_recalled * common_grid_area)
    return area_recalled / area_to_recall if area_to_recall > 0 else np.nan


def entropy(itvls, labels, check_overlap=False):
    """
    Compute the entropy of a set of intervals and their corresponding labels.
    Returns
        The entropy of the intervals and labels in bits.
    """
    # the implementation logic here assumes that the intervals are non-overlapping.
    if check_overlap:
        # If the intervals are sorted by start time, we can just compare consecutive intervals
        # to check for overlaps.
        sorted_itvls = sorted(itvls, key=lambda iv: iv[0])
        for prev, curr in zip(sorted_itvls, sorted_itvls[1:]):
            if prev[1] > curr[0]:
                raise ValueError("Intervals overlap: {} and {}".format(prev, curr))

    # We get label and duration for each segment, aggrecate duration for each label,
    # and compute entropy by assuming a uniform distribution over the union of all intervals provided.
    seg_dur = np.diff(np.array(itvls), axis=1).flatten()
    pi_sum = seg_dur.sum()
    if pi_sum == 0.0:
        return 1.0

    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    label_durs = np.zeros(len(unique_labels))

    for i, duration in enumerate(seg_dur):
        label_durs[inverse_indices[i]] += duration

    pi = label_durs[label_durs > 0]  # Avoid log(0)
    pi_sum = seg_dur.sum()
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))


def make_common_itvls(
    itvls1,
    labels1,
    itvls2,
    labels2,
    # label_condition,
):
    """Label condition is a element of labels2,
    we slice the itvls and labels to get a new sub segmentation
    based on the conditioned on labels2 = label_condition.
    """

    # Strategy: build a new set of common_intervals, and labels of equal length, then do array indexing.
    # Merge boundaries and compute segment durations.
    common_bs = common_boundaries([itvls1, itvls2])
    common_itvls = mir_eval.util.boundaries_to_intervals(common_bs)

    # Find the label at each common boundary.
    gridded_labels1 = np.array(label_at_ts(itvls1, labels1, common_bs[:-1]))
    gridded_labels2 = np.array(label_at_ts(itvls2, labels2, common_bs[:-1]))
    return common_itvls, gridded_labels1, gridded_labels2


def get_conditioned_entropies(itvls, labels, itvls_cond, labels_cond):
    # Get the common intervals and labels
    common_itvls, gridded_labels, gridded_cond_labels = make_common_itvls(
        itvls, labels, itvls_cond, labels_cond
    )
    common_durs = np.diff(common_itvls, axis=1).flatten()

    # Get entropy per condition and store them:
    entropy_of_condition = []
    probabilities_of_condition = []
    labels_of_condition = []
    for cond_l in np.unique(labels_cond):
        # Create a mask where the segments get the conditioned label
        segments_to_keep = gridded_cond_labels == cond_l
        sub_itvls = common_itvls[segments_to_keep]
        sub_labels = gridded_labels[segments_to_keep]
        sub_durs = common_durs[segments_to_keep]

        # Collect
        labels_of_condition.append(cond_l)
        entropy_of_condition.append(entropy(sub_itvls, sub_labels))
        probabilities_of_condition.append(np.sum(sub_durs) / np.sum(common_durs))
    return (
        np.array(entropy_of_condition),
        np.array(probabilities_of_condition),
        np.array(labels_of_condition),
    )


def conditional_entropy(
    itvls,
    labels,
    cond_itvls,
    cond_labels,
):
    """
    Compute the conditional entropy of a set of intervals given a condition.
    """
    # Get the common intervals and labels
    entropy_per_label, prob_per_label, unique_labels = get_conditioned_entropies(
        itvls, labels, cond_itvls, cond_labels
    )
    return entropy_per_label.dot(prob_per_label)


def vmeasure(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
    beta=1.0,
):
    """
    Compute the V-measure score for a reference and estimated labeled segmentation.

    Parameters
    ----------
    ref_itvls : list
        The reference intervals.
    ref_labels : list
        The labels corresponding to the reference intervals.
    est_itvls : list
        The estimated intervals.
    est_labels : list
        The labels corresponding to the estimated intervals.

    Returns
    -------
    precision, recall, f1
        The V-measure.
    """
    recall = 1 - conditional_entropy(
        ref_itvls,
        ref_labels,
        est_itvls,
        est_labels,
    ) / entropy(ref_itvls, ref_labels)
    precision = 1 - conditional_entropy(
        est_itvls,
        est_labels,
        ref_itvls,
        ref_labels,
    ) / entropy(est_itvls, est_labels)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=1.0)


def lmeasure(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
    meet_mode="deepest",
    beta=1.0,
):
    recall = triplet_recall(
        ref_itvls,
        ref_labels,
        est_itvls,
        est_labels,
        meet_mode=meet_mode,
        transitive=True,
    )
    precision = triplet_recall(
        est_itvls,
        est_labels,
        ref_itvls,
        ref_labels,
        meet_mode=meet_mode,
        transitive=True,
    )
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def tmeasure(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
    meet_mode="deepest",
    transitive=False,
):
    """
    Compute the T-measure score for a reference and estimated labeled segmentation.
    Returns
    -------
    precision, recall, f1
        The T-measure.
    """
    # make labels non repeating
    recall = triplet_recall(
        ref_itvls,
        ref_itvls,
        est_itvls,
        est_itvls,
        meet_mode=meet_mode,
        transitive=transitive,
    )
    precision = triplet_recall(
        est_itvls,
        est_itvls,
        ref_itvls,
        ref_itvls,
        meet_mode=meet_mode,
        transitive=transitive,
    )
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=1.0)


def pairwise_clustering(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
    beta=1.0,
):
    """
    Compute the pairwise clustering score for a reference and estimated labeled segmentation.
    """
    recall = pairwise_recall(ref_itvls, ref_labels, est_itvls, est_labels)
    precision = pairwise_recall(est_itvls, est_labels, ref_itvls, ref_labels)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)
