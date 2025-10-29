import frameless_eval as fle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import rel_entr

from .metrics import bmeasure2


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
        ref_bpc, grid_times = ref.prominence_mat(bw=0.8)
        est_bpcs = {key: est.prominence_mat(bw=0.8)[0] for key, est in ests.items()}

        # Setup the objective function, the KL divergence between two BPCs
        # quantized into PMF bins over time
        return ref_bpc.sum(axis=0), est_bpcs, grid_times
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
    # Ensure weights are valid probabilities
    weights = np.maximum(weights, 0)  # Non-negative
    weights = weights / (np.sum(weights) + 1e-10)  # Normalize to sum to 1

    # Combined distribution
    combined = weights @ distributions

    # KL divergence: sum(p * log(p/q))
    kl_div = np.sum(rel_entr(target, combined))

    return kl_div


def scipy_kl_optimization(target_distribution, distributions, verbose=True):
    """
    Solve KL divergence minimization using scipy.optimize.

    Args:
        distributions: Array of shape (num_distributions, distribution_dim) - input probability distributions
        target_distribution: Array of shape (distribution_dim,) - target distribution
        verbose: Whether to print optimization details

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
        kl_div,
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

    # KL divergence
    optimal_kl = kl_div(result.x, distributions, target_distribution)

    return result.x, optimal_combined, optimal_kl
