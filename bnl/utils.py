import mir_eval
import numpy as np

# Suppress mir_eval warnings
mir_eval.hierarchy.validate_hier_intervals = lambda x: None
mir_eval.segment.validate_boundary = lambda x, y, z: None
mir_eval.segment.validate_structure = lambda w, x, y, z: None

def quantize(data, quantize_method='percentile', quant_bins=8):
    data_shape = data.shape
    if quantize_method == 'percentile':
        # Implement percentile quantization
        quantiles = np.percentile(data, np.linspace(0, 100, quant_bins + 1))
        quant_data_flat = np.digitize(data, quantiles) - 1
    elif quantize_method == 'kmeans':
        # Implement k-means quantization
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=quant_bins)
        quant_data_flat = kmeans.fit_predict(data.reshape(-1, 1))
    elif quantize_method is None:
        quant_data_flat = data.flatten()
    else:
        raise ValueError("Invalid quantization method specified.")

    return quant_data_flat.reshape(data_shape)


def laplacian(rec_mat, normalization='random_walk'):
    degree_matrix = np.diag(np.sum(rec_mat, axis=1))
    unnormalized_laplacian = degree_matrix - rec_mat
    if normalization == 'random_walk':
        d_inv = np.diag(1.0 / np.sqrt(np.sum(rec_mat, axis=1)))
        return d_inv @ unnormalized_laplacian @ d_inv
    elif normalization == 'symmetrical':
        d_inv = np.diag(1.0 / np.sum(rec_mat, axis=1))
        return d_inv @ unnormalized_laplacian
    elif normalization is None:
        return unnormalized_laplacian
    else:
        raise ValueError("Invalid normalization type specified.")