import mir_eval
import numpy as np
from collections import defaultdict

from scipy import linalg, stats
from sklearn.cluster import KMeans

from .expand_hier import expand_hierarchy


# Suppress mir_eval warnings
def suppress_mir_eval_warnings():
    mir_eval.hierarchy.validate_hier_intervals = lambda x: None
    mir_eval.segment.validate_boundary = lambda x, y, z: None
    mir_eval.segment.validate_structure = lambda w, x, y, z: None


def quantize(data, quantize_method="percentile", quant_bins=8):
    # method can me 'percentile' 'kmeans'. Everything else will be no quantize
    data_shape = data.shape
    if quantize_method == "percentile":
        bins = [
            np.percentile(data[data > 0], bin * (100.0 / (quant_bins - 1)))
            for bin in range(quant_bins)
        ]
        # print(bins)
        quant_data_flat = np.digitize(data.flatten(), bins=bins, right=False)
    elif quantize_method == "kmeans":
        kmeans_clusterer = KMeans(n_clusters=quant_bins, n_init=50, max_iter=500)
        quantized_non_zeros = kmeans_clusterer.fit_predict(data[data > 0][:, None])
        # make sure the kmeans group are sorted with asending centroid and relabel

        nco = stats.rankdata(kmeans_clusterer.cluster_centers_.flatten())
        # print(kmeans_clusterer.cluster_centers_, nco)
        quantized_non_zeros = np.array([nco[g] for g in quantized_non_zeros], dtype=int)

        quant_data = np.zeros(data.shape)
        quant_data[data > 0] = quantized_non_zeros
        quant_data_flat = quant_data.flatten()
    elif quantize_method is None:
        quant_data_flat = data.flatten()
    else:
        raise ValueError("bad quantize method")

    return quant_data_flat.reshape(data_shape)


def laplacian(rec_mat, normalization="random_walk"):
    degree_matrix = np.diag(np.sum(rec_mat, axis=1))
    unnormalized_laplacian = degree_matrix - rec_mat
    # Compute the Random Walk normalized Laplacian matrix
    if normalization == "random_walk":
        degree_inv = np.linalg.inv(degree_matrix)
        return degree_inv @ unnormalized_laplacian
    elif normalization == "symmetrical":
        sqrt_degree_inv = np.linalg.inv(np.sqrt(degree_matrix))
        return sqrt_degree_inv @ unnormalized_laplacian @ sqrt_degree_inv
    elif normalization is None:
        return unnormalized_laplacian
    else:
        raise NotImplementedError(f"bad laplacian normalization mode: {normalization}")


# from bmcfee/lsd_viz
def _reindex_labels(ref_int, ref_lab, est_int, est_lab):
    # for each estimated label
    #    find the reference label that is maximally overlaps with

    score_map = defaultdict(lambda: 0)

    for r_int, r_lab in zip(ref_int, ref_lab):
        for e_int, e_lab in zip(est_int, est_lab):
            score_map[(e_lab, r_lab)] += max(
                0, min(e_int[1], r_int[1]) - max(e_int[0], r_int[0])
            )

    r_taken = set()
    e_map = dict()

    hits = [(score_map[k], k) for k in score_map]
    hits = sorted(hits, reverse=True)

    while hits:
        cand_v, (e_lab, r_lab) = hits.pop(0)
        if r_lab in r_taken or e_lab in e_map:
            continue
        e_map[e_lab] = r_lab
        r_taken.add(r_lab)

    # Anything left over is unused
    unused = set(est_lab) - set(ref_lab)

    for e, u in zip(set(est_lab) - set(e_map.keys()), unused):
        e_map[e] = u

    return [e_map[e] for e in est_lab]


def reindex(hierarchy):
    new_hier = [hierarchy[0]]
    for i in range(1, len(hierarchy)):
        ints, labs = hierarchy[i]
        labs = _reindex_labels(new_hier[i - 1][0], new_hier[i - 1][1], ints, labs)
        new_hier.append((ints, labs))

    return new_hier


def eigen_gap_scluster(M, k=None, min_k=1):
    # Spectral clustering (scluster) with k groups. If k is None, determine the number of clusters
    # by identifying a significant jump (eigen gap) in the sorted eigenvalues of the Laplacian matrix.
    L = laplacian(M, normalization="random_walk")
    # Assuming L_rw is your random walk normalized Laplacian matrix
    evals, evecs = linalg.eig(L)
    evals = evals.real
    evecs = evecs.real
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    egaps = np.diff(evals)
    T = len(evals)

    # Determine number of clusters using eigen gap heuristic if k is not provided
    if k is None:
        if min_k >= T or max(evals) < 0.1:
            k = T  # Allow singleton group when all eigenvalues are tiny
        else:
            k = np.argmax(egaps[min_k - 1 :]) + min_k

    membership = evecs[:, :k]
    KM = KMeans(n_clusters=k, n_init=50, max_iter=500)
    return KM.fit_predict(membership), k


def slice_matrix(matrix, old_bounds, new_bounds):
    """
    Slice the input matrix so that its grid is defined on the union of old_bounds and new_bounds.
    Each original cell is partitioned according to its fractional overlap with the new grid,
    assuming a uniform distribution within the cell.

    Parameters
    ----------
    matrix : numpy.ndarray
        Original 2D array with shape (n, n) defined over old_bounds intervals.
    old_bounds : array-like
        Original bin boundaries (length n+1).
    new_bounds : array-like
        New boundaries to insert.

    Returns
    -------
    sliced_matrix : numpy.ndarray
        New 2D array with shape (m, m), where m = len(sliced_boundaries)-1, with values
        distributed by fractional area.
    sliced_bounds : numpy.ndarray
        The sorted union of old_bounds and new_bounds.
    """
    # Union of boundaries
    sliced_bounds = np.unique(np.concatenate([old_bounds, new_bounds]))
    m = len(sliced_bounds) - 1
    sliced_matrix = np.zeros((m, m))

    n = len(old_bounds) - 1
    # Loop over original cells
    for i in range(n):
        r0, r1 = old_bounds[i], old_bounds[i + 1]
        # Find indices in sliced_bounds that lie within [r0, r1]
        start_r = np.searchsorted(sliced_bounds, r0, side="left")
        end_r = np.searchsorted(sliced_bounds, r1, side="right") - 1
        for j in range(n):
            c0, c1 = old_bounds[j], old_bounds[j + 1]
            start_c = np.searchsorted(sliced_bounds, c0, side="left")
            end_c = np.searchsorted(sliced_bounds, c1, side="right") - 1

            cell_val = matrix[i, j]
            cell_height = r1 - r0
            cell_width = c1 - c0
            # Distribute the original cell's value into sub-cells by area fraction
            for r in range(start_r, end_r):
                dr = sliced_bounds[r + 1] - sliced_bounds[r]
                frac_r = dr / cell_height
                for c in range(start_c, end_c):
                    dc = sliced_bounds[c + 1] - sliced_bounds[c]
                    frac_c = dc / cell_width
                    sliced_matrix[r, c] += cell_val * frac_r * frac_c
    return sliced_matrix, sliced_bounds


def resample_matrix(matrix, old_bounds, new_bounds):
    """
    Resample a matrix based on new boundary definitions.

    This function takes an input matrix with boundaries defined by old_bounds,
    and resamples it to a new matrix based on new_bounds. The values in the
    new matrix are calculated by summing the corresponding regions from the
    original matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        The input matrix to be resampled. Should be a 2D numpy array.
    old_bounds : array-like
        The boundaries/edges that define the original matrix's bins or cells.
        Length should be number of rows/columns + 1.
    new_bounds : array-like
        The desired boundaries for the resampled matrix.

    Returns
    -------
    numpy.ndarray
        A new matrix with dimensions (len(new_bounds)-1, len(new_bounds)-1),
        containing the summed values from the original matrix based on the
        mapping between old and new boundaries.

    Notes
    -----
    The function assumes that new_bounds values are within the range of old_bounds.
    Each cell (i,j) in the new matrix contains the sum of all cells from the original
    matrix that fall within the corresponding region defined by new_bounds[i:i+2]
    and new_bounds[j:j+2].
    """
    if set(new_bounds) - set(old_bounds):
        matrix, old_bounds = slice_matrix(matrix, old_bounds, new_bounds)

    indices = np.searchsorted(old_bounds, new_bounds)
    new_size = len(new_bounds) - 1
    new_matrix = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            top, bottom = indices[i], indices[i + 1]
            left, right = indices[j], indices[j + 1]
            new_matrix[i, j] = np.sum(matrix[top:bottom, left:right])

    return new_matrix


def gauc(meet_mat_ref, meet_mat_est, agg_mode="frame", transitive=True, window=None):
    """
    Compute ranking recall and normalizer for each query position.

    Parameters:
    -----------
    meet_mat_ref : scipy.sparse matrix
        Reference meet matrix
    meet_mat_est : scipy.sparse matrix
        Estimated meet matrix
    agg_mode : str
        Aggregation mode. 'frame' for frame-wise aggregation, 'triplet' for triplet-wise aggregation.
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
    agg_recall: float
        Aggregated ranking recall according to agg_mode
    q_ranking_recall : numpy.ndarray
        Ranking recall for each query position
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

    q_ranking_recall = np.zeros(n)
    q_ranking_normalizer = np.zeros(n)

    for query in range(n):
        # Get the window around the query
        win_slice = slice(max(0, query - window), min(n, query + window))
        ref_window = meet_mat_ref[query, win_slice].toarray().ravel()
        est_window = meet_mat_est[query, win_slice].toarray().ravel()
        # get the query'th row
        q_window_ref = np.delete(ref_window, query)
        q_window_est = np.delete(est_window, query)
        # count ranking violations
        inversions, normalizer = mir_eval.hierarchy._compare_frame_rankings(
            q_window_ref, q_window_est, transitive=transitive
        )
        q_ranking_recall[query] = (1.0 - inversions / normalizer) if normalizer else 0
        q_ranking_normalizer[query] = normalizer

    if agg_mode == "triplet":
        agg_recall = np.sum(q_ranking_recall) / np.sum(q_ranking_normalizer)
    elif agg_mode == "frame":
        agg_recall = np.mean(q_ranking_recall[np.where(q_ranking_normalizer != 0)])
    else:
        raise ValueError("Invalid aggregation mode specified.")

    return agg_recall, q_ranking_recall, q_ranking_normalizer


def cluster_boundaries(boundaries, novelty, ticks, depth, boundary_time_decimal=3):
    """Convert boundaries with novelty values into hierarchical intervals of specified depth.

    Args:
        boundaries (np.ndarray): Array of boundary indices
        novelty (np.ndarray): Novelty curve values
        ticks (np.ndarray): Time points corresponding to novelty curve in seconds
        depth (int): Maximum number of hierarchical levels

    Returns:
        list: List of interval arrays, one per hierarchical level
    """
    # Determine depth based on unique boundary salience
    depth = min(depth, len(np.unique(novelty[boundaries])))

    # Quantize boundary salience via KMeans
    boundary_salience = quantize(
        novelty[boundaries], quantize_method="kmeans", quant_bins=depth
    )
    rated_boundaries = {
        round(ticks[b], boundary_time_decimal): s
        for b, s in zip(boundaries, boundary_salience)
    }

    # Create hierarchical intervals based on salience thresholds
    intervals = []
    for l in range(depth):
        salience_thresh = depth - l
        boundaries_at_level = [
            b for b in rated_boundaries if rated_boundaries[b] >= salience_thresh
        ]
        intervals.append(mir_eval.util.boundaries_to_intervals(boundaries_at_level))
    return intervals


def pad_itvls(ref_itvls, est_itvls):
    """Make sure ref hier and est hier are the same length by padding the shorter one"""
    max_length = max(ref_itvls[-1][-1, -1], est_itvls[-1][-1, -1])
    # Iterate over all the levels and make the last segment end at max_length
    for i in range(len(ref_itvls)):
        ref_itvls[i][-1, 1] = max_length
    for i in range(len(est_itvls)):
        est_itvls[i][-1, 1] = max_length

    return ref_itvls, est_itvls


def fill_out_anno(anno, new_duration):
    """
    Fill out an annotation object by adding "NL" (No Label) segments at the beginning and end if necessary.

    Parameters:
    ----------
    anno : jams Annotation
        The annotation object to fill out.
    new_duration : float
        The max time to fill out to.

    Returns:
    -------
    Annotation
        The filled out annotation object with added "NL" segments if needed.

    Notes:
    -----
    This function ensures the annotation covers the entire time range from 0 to the last frame time
    by adding "NL" segments at:
    1. The beginning (from time 0 to annotation start) if the annotation doesn't start at time 0
    2. The end (from annotation end to last frame time) if the annotation ends before the last frame
    """
    anno_start_time = anno.data[0].time
    anno_end_time = anno.data[-1].time + anno.data[-1].duration
    anno.duration = new_duration

    if anno_end_time > anno.duration:
        raise ValueError(
            f"Annotation end time {anno_end_time} exceeds new duration {anno.duration}"
        )
    if anno_start_time != 0:
        anno.append(value="NL", time=0, duration=anno_start_time, confidence=1)
    if anno_end_time < anno.duration:
        anno.append(
            value="NL",
            time=anno_end_time,
            duration=anno.duration - anno_end_time,
            confidence=1,
        )

    return anno
