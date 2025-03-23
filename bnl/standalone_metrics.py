import mir_eval
import numpy as np


def meet(hier_itvls, hier_labels, u, v, mode="deepest", compare_fn=np.equal):
    """
    Compute the meeting point for a list of given time pairs (u, v) and mode.
    This function calculates the meet matrix based at the time points and with mode.

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
        return len(lvl_meet) - np.argmax(lvl_meet[::-1]) if any(lvl_meet) else 0
    elif mode == "mono":
        # Find the idx of the first False value, len(lvl_meet) if all are True
        return np.argmax(lvl_meet == False) if not all(lvl_meet) else len(lvl_meet)
    else:
        raise ValueError(f"Unknown meeting mode: {mode}.\n Use 'deepest' or 'mono'.")


def labels_at_t(hier_itvls, hier_labels, t):
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


def relevance_at_t(
    hier_itvls, hier_labels, t, meet_mode="deepest", compare_fn=np.equal
):
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
    all_bs = np.sort(np.unique(np.concatenate(hier_itvls)))
    # get the meet at time t, all_bs[:-1], thats the relevance of t against each segemnt.
    rel_val = np.vectorize(
        lambda u: float(
            meet(hier_itvls, hier_labels, t, u, meet_mode, compare_fn=compare_fn)
        )
    )(all_bs[:-1])

    return mir_eval.util.boundaries_to_intervals(all_bs), rel_val


def triplet_recall_at_t(
    ref_hier_itvls,
    ref_hier_labels,
    est_hier_itvls,
    est_hier_labels,
    t,
    meet_mode="deepest",
    window=0,
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
    pass


def triplet_recall(
    ref_hier_itvls,
    ref_hier_labels,
    est_hier_itvls,
    est_hier_labels,
    meet_mode="deepest",
    window=0,
    transitive=True,
    debug=0,
):
    pass


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
    pass


def entropy(itvls, labels):
    """
    Compute the entropy of a set of intervals and their corresponding labels.

    Parameters
    ----------
    itvls : list
        The intervals.
    labels : list
        The labels corresponding to the intervals.

    Returns
    -------
    float
        The entropy of the intervals and labels.
    """
    pass


def conditional_entropy(itvls, labels, condition_itvls, condition_labels):
    """
    Compute the conditional entropy of a set of intervals and their corresponding labels.

    Parameters
    ----------
    itvls : list
        The intervals.
    labels : list
        The labels corresponding to the intervals.
    condition_itvls : list
        The conditioning intervals.
    condition_labels : list
        The labels corresponding to the conditioning intervals.

    Returns
    -------
    float
        The conditional entropy of the intervals and labels.
    """
    return None


def vmeasure(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
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
    pass


def lmeasure(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
):
    """
    Compute the L-measure score for a reference and estimated labeled segmentation.

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
        The L-measure.
    """
    pass


def tmeasure(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
    window=15,
    transitive=False,
):
    """
    Compute the T-measure score for a reference and estimated labeled segmentation.

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
        The T-measure.
    """
    pass


def pair_clustering(
    ref_itvls,
    ref_labels,
    est_itvls,
    est_labels,
):
    """
    Compute the pairwise clustering score for a reference and estimated labeled segmentation.

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
        The pairwise clustering score.
    """
    pass
