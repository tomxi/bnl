import mir_eval, itertools, collections
import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix


def pairwise(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0):
    # make sure est the same lenght as ref
    aligned_hiers = align_hier([ref_itvls], [ref_labels], [est_itvls], [est_labels])
    # Make common grid
    common_itvls, ref_labs, est_labs = make_common_itvls(*aligned_hiers)

    # Get the segment durations and use as weights
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    # Build label agreement maps (meet matrix)
    meet_ref = _meet(ref_labs)
    meet_est = _meet(est_labs)
    meet_both = meet_ref * meet_est

    seg_pair_size = np.outer(seg_dur, seg_dur)
    ref_sig_pair_size = np.sum(meet_ref * seg_pair_size)
    est_sig_pair_size = np.sum(meet_est * seg_pair_size)
    intersection_size = np.sum(meet_both * seg_pair_size)

    precision = intersection_size / est_sig_pair_size
    recall = intersection_size / ref_sig_pair_size
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def vmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0):
    # make sure est the same lenght as ref
    aligned_hiers = align_hier([ref_itvls], [ref_labels], [est_itvls], [est_labels])
    # Make common grid
    common_itvls, ref_labs, est_labs = make_common_itvls(*aligned_hiers)
    # Get the segment durations
    seg_dur = np.diff(common_itvls, axis=1).flatten()

    # Get the contingency matrix and normalize
    contingency, _, _ = _weighted_contingency(ref_labs, est_labs, seg_dur)
    contingency = contingency / np.sum(seg_dur)

    # Compute the marginals
    p_est = contingency.sum(axis=0)
    p_ref = contingency.sum(axis=1)

    # H(true | prediction) = sum_j P[estimated = j] *
    # sum_i P[true = i | estimated = j] log P[true = i | estimated = j]
    # entropy sums over axis=0, which is true labels

    true_given_est = p_est.dot(stats.entropy(contingency, base=2))
    pred_given_ref = p_ref.dot(stats.entropy(contingency.T, base=2))

    # Normalize conditional entropy by marginal entropy
    z_ref = stats.entropy(p_ref, base=2)
    z_est = stats.entropy(p_est, base=2)
    r = (1.0 - true_given_est / z_ref) if z_ref > 0 else 0
    p = (1.0 - pred_given_ref / z_est) if z_est > 0 else 0
    return p, r, mir_eval.util.f_measure(p, r, beta=beta)


def lmeasure(ref_itvls, ref_labels, est_itvls, est_labels, beta=1.0, mono=False):
    # make sure est the same lenght as ref
    aligned_hiers = align_hier(ref_itvls, ref_labels, est_itvls, est_labels)
    # Make common grid
    common_itvls, ref_labs, est_labs = make_common_itvls(*aligned_hiers)
    seg_dur = np.diff(common_itvls, axis=1).flatten()
    meet_ref = _meet(ref_labs, mono=mono)
    meet_est = _meet(est_labs, mono=mono)

    recall = triplet_recall(meet_ref, meet_est, seg_dur)
    precision = triplet_recall(meet_est, meet_ref, seg_dur)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def triplet_recall(meet_ref, meet_est, seg_dur):
    per_segment_recall = []
    for seg_idx in range(len(seg_dur)):
        # get per segment recall
        per_segment_recall.append(
            _segment_triplet_recall(meet_ref, meet_est, seg_idx, seg_dur)
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


# def bmeasure(ref_itvls, est_itvls, beta=1.0, trim=False):
# # make sure est the same lenght as ref
# aligned_hiers = align_hier(ref_itvls, None, est_itvls, None)

# return p, r, mir_eval.util.f_measure(p, r, beta=beta)


def boundary_counts(hier_itvls: list, trim=True):
    """
    Compute the boundary counts for a hierarchy of intervals.

    """
    # Get the boundaries of the intervals
    bs_counts = collections.Counter()
    for itvls in hier_itvls:
        new_bs = mir_eval.util.intervals_to_boundaries(itvls)
        if trim:
            new_bs = new_bs[1:-1]
        bs_counts.update(new_bs)
    return bs_counts


def query_salience(bs_sals: dict, query_bs: np.ndarray, match_tolerance_window=0.5):
    """
    Query the salience of a rated boundary.
    bs_sals: dict of rated boundaries
    query_bs: numpy array of query boundaries

    returns: a numpy array of salience.
    """
    bs = np.asarray(list(bs_sals.keys()))
    hits = mir_eval.util.match_events(query_bs, bs, window=match_tolerance_window)

    # q, b are match indices
    queried_sal = np.zeros(len(query_bs))
    for q_idx, b_idx in hits:
        queried_sal[q_idx] = bs_sals[bs[b_idx]]
    return queried_sal


def rated_bs_recall(ref_sal: np.ndarray, est_sal: np.ndarray):
    # Any pairs involving a zero salience are ignored, this is to have a control on the included pairs.
    # get where both saliences are non zero
    valid_mask = (ref_sal > 0) * (est_sal > 0)
    hit_recall = (
        np.count_nonzero(valid_mask) / len(ref_sal) if len(ref_sal) > 0 else 1.0
    )

    inversions, valid_pairs = mir_eval.hierarchy._compare_frame_rankings(
        ref_sal[valid_mask],
        est_sal[valid_mask],
        transitive=True,
    )
    rank_recall = (
        (valid_pairs - inversions) / valid_pairs if valid_pairs > 0 else np.nan
    )

    return hit_recall, rank_recall


def bmeasure(ref_itvls, est_itvls, window=0.5, beta=1.0, trim=True):
    ref_bs_sal = boundary_counts(ref_itvls, trim=trim)
    est_bs_sal = boundary_counts(est_itvls, trim=trim)
    ref_sal = np.array(list(ref_bs_sal.values()))
    est_sal = np.array(list(est_bs_sal.values()))

    # Get the salience at the other annotation's boundaries
    est_sal_at_ref_bs = query_salience(
        est_bs_sal, np.array(list(ref_bs_sal.keys())), match_tolerance_window=window
    )
    ref_sal_at_est_bs = query_salience(
        ref_bs_sal, np.array(list(est_bs_sal.keys())), match_tolerance_window=window
    )

    # Compute the recall and precision
    hit_r, rank_r = rated_bs_recall(ref_sal, est_sal_at_ref_bs)
    hit_p, rank_p = rated_bs_recall(est_sal, ref_sal_at_est_bs)

    # we use the harmonic mean of the rank and hit measure to get the bmeasure.
    recall = hit_r if rank_r is np.nan else mir_eval.util.f_measure(hit_r, rank_r)
    precision = hit_p if rank_p is np.nan else mir_eval.util.f_measure(hit_p, rank_p)
    return precision, recall, mir_eval.util.f_measure(precision, recall, beta=beta)


def labels_at_ts(hier_itvls: list, hier_labels: list, ts: np.ndarray):
    """
    get label at ts for all levels in a hierarchy
    """
    results = []
    for itvls, labs in zip(hier_itvls, hier_labels):
        result = _label_at_ts(itvls, labs, ts)
        results.append(result)
    return results


def make_common_itvls(
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


def _common_boundaries(list_of_itvls):
    # Get the boundaries of both sets of intervals
    bs = set()
    for itvls in list_of_itvls:
        bs = bs.union(set(mir_eval.util.intervals_to_boundaries(itvls)))
    return np.array(sorted(bs))


def _segment_triplet_recall(meet_ref, meet_est, seg_idx, seg_dur, transitive=True):
    # given a segment idx, their relevance against each other segment.
    ref_rel_againt_seg_i = meet_ref[seg_idx, :]
    est_rel_againt_seg_i = meet_est[seg_idx, :]

    # use count inversions to get normalizer and number of inversions
    inversions, normalizer = _compare_segment_rankings(
        ref_rel_againt_seg_i,
        est_rel_againt_seg_i,
        ref_w=seg_dur,
        we=seg_dur,
        transitive=transitive,
    )
    return 1.0 - inversions / normalizer if normalizer > 0 else np.nan


def _label_at_ts(itvls, labels, ts):
    """
    Assign labels to timestamps using interval boundaries.

    Parameters
    ----------
    itvls : np.ndarray
        An array of shape (n, 2) representing intervals.
    labels : array_like
        An array-like object of labels corresponding to each interval.
    ts : array_like
        Timestamps to be labeled.

    Returns
    -------
    np.ndarray
        An array of labels corresponding to each timestamp in ts.
    """
    boundaries = mir_eval.util.intervals_to_boundaries(itvls)
    labels = np.asarray(labels)
    extended = np.concatenate([labels, [labels[-1]]])  # Extend last label
    return extended[np.searchsorted(boundaries, np.atleast_1d(ts), side="right") - 1]


def _count_weighted_inversions(a, wa, b, wb):
    """
    Count weighted inversions between two arrays.
    An inversion is any pair (i, j) with a[i] >= b[j],
    contributing wa[i] * wb[j] to the sum.
    """
    ua, inv_a = np.unique(a, return_inverse=True)
    wa_sum = np.bincount(inv_a, weights=wa)
    ub, inv_b = np.unique(b, return_inverse=True)
    wb_sum = np.bincount(inv_b, weights=wb)

    inversions = 0.0
    i = j = 0
    while i < len(ua) and j < len(ub):
        if ua[i] < ub[j]:
            i += 1
        else:
            inversions += np.sum(wa_sum[i:]) * wb_sum[j]
            j += 1
    return inversions


def _compare_segment_rankings(ref, est, ref_w=None, est_w=None, transitive=False):
    """
    Compute weighted ranking disagreements between two lists.

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Reference ranked list.
    est : np.ndarray, shape=(n,)
        Estimated ranked list.
    ref_w : np.ndarray, shape=(n,), optional
        Weights for ref (default: ones).
    we : np.ndarray, shape=(n,), optional
        Weights for est (default: ones).
    transitive : bool, optional
        If True, compare all pairs of distinct ref levels;
        if False, compare only adjacent levels.

    Returns
    -------
    inversions : float
        Weighted inversion count: sum_{(i,j) in pairs} [inversions between est slices].
    normalizer : float
        Total weighted number of pairs considered.
    """
    n = len(ref)
    if ref_w is None:
        ref_w = np.ones(n)
    if est_w is None:
        est_w = np.ones(n)

    # The algo starts by sorting everything by ref's values for easy inversion counting.
    idx = np.argsort(ref)
    ref, est = ref[idx], est[idx]
    ref_w, est_w = ref_w[idx], est_w[idx]

    # Get the unique levels of values and their positions in the sorted array
    levels, pos = np.unique(ref, return_index=True)
    pos = list(pos) + [len(ref)]

    # For each group of segments that has the same level/label value, we get the summed weights.
    level_groups = {
        lvl: slice(start, end) for lvl, start, end in zip(levels, pos[:-1], pos[1:])
    }
    ref_level_weights = {lvl: np.sum(ref_w[level_groups[lvl]]) for lvl in levels}

    if transitive:
        level_pairs = itertools.combinations(levels, 2)
    else:
        level_pairs = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]

    # Create two independent iterators over level_pairs.
    level_pairs, level_pairs_counts = itertools.tee(level_pairs)
    normalizer = float(
        sum(ref_level_weights[i] * ref_level_weights[j] for i, j in level_pairs_counts)
    )
    if normalizer == 0:
        return 0.0, 0.0

    # We already sorted by ref array, so we count inversions now.
    inversions = sum(
        _count_weighted_inversions(
            est[level_groups[l1]],
            est_w[level_groups[l1]],
            est[level_groups[l2]],
            est_w[level_groups[l2]],
        )
        for l1, l2 in level_pairs
    )
    return inversions, float(normalizer)


def _weighted_contingency(ref_labels, est_labels, durations):
    """
    Build a weighted contingency matrix.
    Each cell (i,j) sums the durations for frames with ref label i and est label j.
    """
    ref_classes, ref_idx = np.unique(ref_labels, return_inverse=True)
    est_classes, est_idx = np.unique(est_labels, return_inverse=True)
    contingency = coo_matrix(
        (durations, (ref_idx, est_idx)),
        shape=(len(ref_classes), len(est_classes)),
        dtype=np.float64,
    ).toarray()
    return contingency, ref_classes, est_classes


def align_hier(ref_itvls, ref_labels, est_itvls, est_labels):
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
