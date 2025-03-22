import mir_eval
import numpy as np


def meet(hier, time_pairs, mode="deepest", compare_fn=np.greater):
    """
    Compute the meeting point for a list of given time pairs (u, v) and mode.
    This function calculates the meet matrix based at the time points and with mode.

    Parameters
    ----------
    hier: list of list of (itvls, labels)
        The hierarchy
    time_pairs : array-like [(u1, v1), (u2, v2), ...]
        The time pairs to consider.
    mode : str
        The meeting mode to use.
        Options are "deepest", "mono", or "mean".
    Returns
    -------
    np.array
        The relevance calculated at the specified time pairs.
        The degree of relevance, or depth of meet.
    """
    if mode == "deepest":
        return None
    elif mode == "mono":
        return None
    elif mode == "mean":
        return None
    else:
        raise ValueError(
            f"Unknown meeting mode: {mode}.\n Use 'deepest', 'mono', or 'mean'."
        )


def relevance_at_t(hier, t, meet_mode="deepest"):
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
        Options are "deepest", "mono", or "mean".
    Returns
    -------
    itvls: list of pairs,
        The intervals and the relevance values.
    relevances: list of values.
    """
    pass


def triplet_recall_at_t(
    ref_hier,
    est_hier,
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
    ref_hier,
    est_hier,
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
    entropy_value = 0.0
    return entropy_value


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
