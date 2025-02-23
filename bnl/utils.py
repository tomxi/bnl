import mir_eval
import numpy as np
from collections import defaultdict

from scipy import linalg
from sklearn.cluster import KMeans


# Suppress mir_eval warnings
def suppress_mir_eval_warnings():
    mir_eval.hierarchy.validate_hier_intervals = lambda x: None
    mir_eval.segment.validate_boundary = lambda x, y, z: None
    mir_eval.segment.validate_structure = lambda w, x, y, z: None


def quantize(data, quantize_method="percentile", quant_bins=8):
    data_shape = data.shape
    if quantize_method == "percentile":
        # Implement percentile quantization
        quantiles = np.percentile(data, np.linspace(0, 100, quant_bins + 1))
        quant_data_flat = np.digitize(data, quantiles) - 1
    elif quantize_method == "kmeans":
        # Implement k-means quantization
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=quant_bins)
        quant_data_flat = kmeans.fit_predict(data.reshape(-1, 1))
    elif quantize_method is None:
        quant_data_flat = data.flatten()
    else:
        raise ValueError("Invalid quantization method specified.")

    return quant_data_flat.reshape(data_shape)


def laplacian(rec_mat, normalization="random_walk"):
    degree_matrix = np.diag(np.sum(rec_mat, axis=1))
    unnormalized_laplacian = degree_matrix - rec_mat
    if normalization == "random_walk":
        d_inv = np.diag(1.0 / np.sqrt(np.sum(rec_mat, axis=1)))
        return d_inv @ unnormalized_laplacian @ d_inv
    elif normalization == "symmetrical":
        d_inv = np.diag(1.0 / np.sum(rec_mat, axis=1))
        return d_inv @ unnormalized_laplacian
    elif normalization is None:
        return unnormalized_laplacian
    else:
        raise ValueError("Invalid normalization type specified.")


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


def _eigen_gap_scluster(M, k=None, min_k=1):
    # scluster with k groups. default is eigen gap.
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


def _resample_matrix(matrix, old_bounds, new_bounds):
    """Resample the given matrix based on new boundaries."""
    indices = np.searchsorted(old_bounds, new_bounds)
    new_size = len(new_bounds) - 1
    new_matrix = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            top, bottom = indices[i], indices[i + 1]
            left, right = indices[j], indices[j + 1]
            new_matrix[i, j] = np.sum(matrix[top:bottom, left:right])

    return new_matrix
