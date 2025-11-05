import frameless_eval as fle
import numpy as np
import pandas as pd
from mir_eval.segment import detection as me_hr
from scipy.optimize import minimize
from scipy.stats import entropy

from .core import MultiSegment
from .metrics import bmeasure2
from .ops import bs2uv, build_time_grid, combine_ms

# region: scipy optimizations


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
    combined = distributions @ weights

    # Normalize distribution
    combined += 1e-12  # add a small number to avoid log(p/q) where q is zero
    target += 1e-12
    combined /= np.sum(combined)
    target /= np.sum(target)

    # KL divergence: sum(p * log(p/q)) entropy takes care of p=0, it returns 0 for that bin.
    kl_div = entropy(target, combined)

    return kl_div


def mse(weights, distributions, target):
    # Combined distribution
    combined = distributions @ weights

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
    combined = distributions @ weights

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


def scipy_optimize(target_distribution, distributions, verbose=False, obj_fn=kl_div):
    """
    Solve KL divergence minimization using scipy.optimize.

    Args:
        distributions: Array of shape (distribution_dim, num_distributions)
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
    dist_dim, num_distributions = distributions.shape

    # Initial guess: uniform weights
    initial_weights = np.ones(num_distributions) / num_distributions

    # constrain the weights to sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

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

    return result.x


# endregion: scipy optimization


# region: relevance functions


def relevance_h2h(ref, ests, metric="b15") -> pd.Series:
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "b30", "b05", "t", "l", "l-exp"
    # Monocasting and label decoding should be done before hand.
    # Let's do the simple case of having existing metrics to compute relevance.

    # h2h should check for boundary monotonicity. and raise error if not monotonic.
    if not ref.has_monotonic_bs():
        raise ValueError("Reference must have monotonic boundaries.")
    for est in ests.values():
        if not est.has_monotonic_bs():
            raise ValueError("Estimates must all have monotonic boundaries.")

    rel = pd.Series(index=ests.keys(), dtype=float, name=metric)

    # For B-measure and T-measure, we use bc objects.
    if metric[0] == "b":
        window = int(metric[1:]) * 0.1
        ref_bc = ref.contour("depth").level("unique")
        for key, est in ests.items():
            est_bc = est.align(ref).contour("depth").level("unique")
            rel[key] = bmeasure2(ref_bc, est_bc, window=window)[2]  # Take the f-score

    # For L-measure, check if we want to do hierarchy label expansion.
    elif metric[0] == "l":
        if metric == "l-exp":
            ref = ref.expand_labels()
        for key, est in ests.items():
            if metric == "l-exp":
                est = est.align(ref).expand_labels()
            rel[key] = fle.lmeasure(ref.itvls, ref.labels, est.itvls, est.labels)[2]

    # For T-measure, let's do L-measure with unique labeling.
    elif metric == "t":
        ref = ref.scrub_labels(None)
        for key, est in ests.items():
            est = est.align(ref).scrub_labels(None)
            rel[key] = fle.lmeasure(ref.itvls, ref.labels, est.itvls, est.labels)[2]

    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return rel


def relevance_h2f(ref, est, metric="bpc", obj_fn=js_div) -> pd.Series:
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "bpc", "lam"
    # right now the gridtime is predefined: for bpc it's 0.1 second in build_time_grid
    # for lam it's 0.5 second for lam

    if metric == "bpc":
        # pick a sampling rate that gives me about less than 20k points, and no finer than 0.1 secs
        frame_size = max(0.1, ref.duration / 20000)
        # print(f"Using frame size: {frame_size}")
        time_grid = build_time_grid(ref, frame_size)
        y = ref.contour("prob").bpc(bw=0.5, time_grid=time_grid)

        est = est.align(ref)
        if len(est) == 0:
            raise ValueError("No valid estimated layers found.")
        x = est.bpcs(bw=0.5, time_grid=time_grid)

    elif metric == "lam":
        # pick a sampling rate that gives me about 20k points
        frame_size = max(0.1, ref.duration / 200)
        # print(f"Using frame size: {frame_size}")
        time_grid = build_time_grid(ref, frame_size)
        sample_points = bs2uv(time_grid)
        # should I use depth or prob for ref lam?
        ref_lam_values = ref.expand_labels().lam(strategy="depth").sample(sample_points)

        est_lams = {layer.name: layer.lam_pdf.sample(sample_points) for layer in est.align(ref)}
        y = pd.Series(ref_lam_values, index=sample_points.T.tolist())
        x = pd.DataFrame(est_lams, index=sample_points.T.tolist())
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    # run scipy optimize
    weights = scipy_optimize(y, x, obj_fn=obj_fn)
    return pd.Series(weights, index=x.columns, name=metric)


def relevance_f2f(ref_layer, est, metric="hr15") -> pd.Series:
    """
    metric can be "hr05", "hr15", "hr30", "v", "pfc"
    """
    rel = pd.Series(index=est.layer_names, dtype=float, name=metric)
    est = est.align(ref_layer)
    if metric[:2] == "hr":
        try:
            window = int(metric[2:]) * 0.1
        except ValueError as e:
            raise ValueError(f"{metric} has invalid window string.") from e
        for est_layer in est:
            # record the relevance of each layer with respect to the reference
            rel[est_layer.name] = me_hr(ref_layer.itvls, est_layer.itvls, window=window)[2]
    elif metric == "v":
        for est_layer in est:
            rel[est_layer.name] = fle.vmeasure(
                ref_layer.itvls, ref_layer.labels, est_layer.itvls, est_layer.labels
            )[2]
    elif metric == "pfc":
        for est_layer in est:
            rel[est_layer.name] = fle.pairwise(
                ref_layer.itvls, ref_layer.labels, est_layer.itvls, est_layer.labels
            )[2]
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return rel


# endregion: relevance functions


# region: compatibility diagrams


def comp_diag_h2h(
    refs: dict[str, MultiSegment], ests: dict[str, MultiSegment], metric="b15"
) -> pd.DataFrame:
    """
    columns:refs, rows: ests
    """
    rels = []
    for name, ref in refs.items():
        r = relevance_h2h(ref, ests, metric=metric)
        r.name = name
        rels.append(r)

    df = pd.concat(rels, axis=1)
    df.name = metric
    return df


def comp_diag_h2f(
    refs: dict[str, MultiSegment], ests: dict[str, MultiSegment], metric="bpc"
) -> pd.DataFrame:
    rels = []

    for name, ref in refs.items():
        est = combine_ms(ests, ignore_names=(name))
        r = relevance_h2f(ref, est, metric=metric)
        r.name = name
        rels.append(r)

    df = pd.concat(rels, axis=1)
    df.name = metric
    return df


def comp_diag_f2f(ref: MultiSegment, est: MultiSegment, metric="hr15") -> pd.DataFrame:
    rels = []
    for ref_layer in ref:
        r = relevance_f2f(ref_layer, est, metric=metric)
        r.name = ref_layer.name
        rels.append(r)

    df = pd.concat(rels, axis=1)
    df.name = metric
    return df


# endregion: compatibility diagrams
