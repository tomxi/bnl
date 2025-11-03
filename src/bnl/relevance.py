import frameless_eval as fle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import entropy

from .metrics import bmeasure2
from .ops import bs2uv, build_time_grid


def relevance(ref, ests, metric="b30", debug=False):
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "b30", "b05", "t", "l", "l-mono-lam", "l-mono-1layer"
    # Let's do the simple case of having existing metrics to compute relevance.
    rel = dict()

    # For B-measure and T-measure, we need to make sure that they are monotonic boundaries.
    if metric[0] == "b":
        ref_bc = ref.contour("depth").level("unique")
        if debug:
            ref_bc.plot().show()
        for key, est in ests.items():
            est_bc = (
                est.align(ref).contour("prob").clean("kde", bw=0.5).level("mean_shift", bw=0.15)
            )
            window = int(metric[1:]) * 0.1
            rel[key] = bmeasure2(ref_bc, est_bc, window=window)[2]  # Take the f-score
            if debug:
                est_bc.plot().show()

    # For L-measure, we need expanded labeling... and no monotonicity requirement for now.
    elif metric == "l":
        ref = ref.expand_labels()
        if debug:
            ref.plot().show()
        for key, est in ests.items():
            est = est.align(ref)
            if debug:
                est.plot().show()
            rel[key] = fle.lmeasure(ref.itvls, ref.labels, est.itvls, est.labels)[2]

    # L-measure on monotonicity casted and label decoding.
    elif metric[:6] == "l-mono":
        ref = ref.expand_labels()
        if debug:
            ref.plot().show()
        for key, est in ests.items():
            est = est.align(ref)
            labeling_strat = metric[7:]
            mono_est = (
                est.contour("prob")
                .clean("kde", bw=0.5)
                .level("mean_shift", bw=0.15)
                .to_ms(strategy=labeling_strat, ref_ms=est)
            )
            if debug:
                mono_est.plot().show()
            rel[key] = fle.lmeasure(ref.itvls, ref.labels, mono_est.itvls, mono_est.labels)[2]

    # For T-measure, we need unique leveling, and monotonic casting.
    elif metric == "t":
        ref_ms = ref.contour("depth").level("unique").to_ms()
        if debug:
            ref_ms.plot().show()
        for key, est in ests.items():
            est_ms = (
                est.align(ref)
                .contour("prob")
                .clean("kde", bw=0.5)
                .level("mean_shift", bw=0.15)
                .to_ms()
            )
            rel[key] = fle.lmeasure(ref_ms.itvls, ref_ms.labels, est_ms.itvls, est_ms.labels)[2]
            if debug:
                est_ms.plot().show()

    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return pd.Series(rel, name=metric)


def relevance_per_level(ref, ests, metric="bpc_combo", debug=False):
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "hr", "v", "bpc_combo", "lam_combo"
    if metric == "bpc_combo":
        # Setup est_bpcs and ref_bpc.
        ref_bpc, ref_layer_ids, grid_times = ref.prominence_mat(bw=0.5)

        est_bpcs = []
        est_labels = []
        for est_key in ests:
            est_bpc, est_layer_ids, _ = (
                ests[est_key].align(ref).prominence_mat(bw=0.5, grid_times=grid_times)
            )
            est_bpcs.extend(est_bpc)
            est_labels.extend(est_layer_ids)

        # Setup the objective function, the KL divergence between two BPCs
        # quantized into PMF bins over time
        if len(est_bpcs) == 0:
            raise ValueError("No estimated boundaries found.")
        return ref_bpc.mean(axis=0), np.asarray(est_bpcs), est_labels, grid_times
    elif metric == "lam_combo":
        # Setup est_lams and ref_lam.
        # setup 0.5 second grid
        grid_times = build_time_grid(ref, 0.5)
        sample_points = bs2uv(grid_times)
        ref_lam_values = ref.expand_labels().lam(strategy="depth").sample(sample_points)

        est_lam_values = []
        est_labels = []
        for est_key in ests:
            est = ests[est_key].align(ref)
            for layer in est:
                est_lam_values.append(layer.lam_pdf.sample(sample_points))
                est_labels.append(est.name + ":" + layer.name)
        return ref_lam_values.astype(float), np.asarray(est_lam_values), est_labels, grid_times
    elif metric == "v":
        pass
    elif metric == "hr":
        pass
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return None


def kl_div(weights, distributions, target):
    """
    KL divergence objective function for scipy.optimize.

    Args:
        weights: Array of shape (num_distributions,) - mixing weights
        distributions: Array of shape (num_distributions, distribution_dim) - input distributions
        target: Array of shape (distribution_dim,) - target distribution

    Returns:
        KL divergence value (float)
    """
    # Combined distribution
    combined = weights @ distributions

    # Normalize distribution
    combined += 1e-12
    target += 1e-12
    combined /= np.sum(combined)
    target /= np.sum(target)

    # KL divergence: sum(p * log(p/q))
    kl_div = entropy(target, combined)

    return kl_div


def mse(weights, distributions, target):
    # Combined distribution
    combined = weights @ distributions

    # Normalize distribution
    combined /= np.sum(combined)
    target /= np.sum(target)

    return np.sum((combined - target) ** 2)


def js_div(weights, distributions, target):
    """
    Jensen-Shannon Divergence (JSD) objective function.

    Computes JSD(target || combined), where 'combined' is the
    weighted mixture of the input 'distributions'.

    Args:
        weights: Array of shape (num_distributions,) - mixing weights
        distributions: Array of shape (num_distributions, distribution_dim) - input distributions
        target: Array of shape (distribution_dim,) - target distribution

    Returns:
        Jensen-Shannon Divergence value (float)
    """

    # Combined distribution
    combined = weights @ distributions

    # Normalize distribution
    combined /= np.sum(combined)
    target /= np.sum(target)

    # Define the average distribution 'A'
    # A = 0.5 * (P + M), where P is 'target'
    avg_dist = 0.5 * (target + combined)

    # Compute the two KL components
    # D_KL(P || A)
    kl_p_a = entropy(target, avg_dist)

    # D_KL(M || A)
    kl_m_a = entropy(combined, avg_dist)

    # Compute the JSD
    jsd = 0.5 * kl_p_a + 0.5 * kl_m_a

    return jsd


def scipy_optimize(target_distribution, distributions, verbose=True, obj_fn=kl_div):
    """
    Solve KL divergence minimization using scipy.optimize.

    Args:
        distributions: Array of shape (num_distributions, distribution_dim)
         - input probability distributions
        target_distribution: Array of shape (distribution_dim,)
         - target distribution
        verbose: Whether to print optimization details
        obj_fn: Objective function to minimize (default: KL divergence)

    Returns:
        optimal_weights: Array of shape (num_distributions,) - optimal mixing weights
        optimal_combined: Array of shape (distribution_dim,) - optimally combined distribution
        optimal_kl: Float - optimal KL divergence value
    """
    num_distributions, dist_dim = distributions.shape

    # Initial guess: uniform weights
    initial_weights = np.ones(num_distributions) / num_distributions

    # Define constraints for scipy
    constraints = [
        # Sum of weights = 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1) for _ in range(num_distributions)]

    if verbose:
        print(f"Number of distributions: {num_distributions}")
        print(f"Distribution dimension: {dist_dim}")

    # Solve optimization problem
    result = minimize(
        obj_fn,
        initial_weights,
        args=(distributions, target_distribution),
        bounds=bounds,
        constraints=constraints,
        options={"disp": verbose, "maxiter": 1000},
    )

    if not result.success:
        print(f"Optimization warning: {result.message}")

    # Combined distribution
    optimal_combined = result.x @ distributions

    # KL/JS divergence
    optimal_div = obj_fn(result.x, distributions, target_distribution)

    return result.x, optimal_combined, optimal_div
