import re
import warnings
from dataclasses import dataclass

import frameless_eval as fle
import numpy as np
import pandas as pd
from mir_eval.segment import detection as me_hr
from scipy import stats
from scipy.optimize import minimize
from scipy.special import rel_entr

from .core import MultiSegment, Segment
from .metrics import bmeasure3
from .ops import bs2uv, build_time_grid, combine_ms, common_itvls, filter_named_ms

# region: scipy optimizations


def js_div(weights, distributions, target, sample_weights):
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

    # Define the average distribution 'A'
    # A = 0.5 * (P + M), where P is 'target'
    avg_dist = 0.5 * (target + combined)

    if sample_weights is None:
        sample_weights = np.ones(distributions.shape[1]) / distributions.shape[1]
    # Compute the two KL components
    # D_KL(P || A)
    kl_p_a = rel_entr(target, avg_dist) @ sample_weights

    # D_KL(M || A)
    kl_m_a = rel_entr(combined, avg_dist) @ sample_weights

    # Compute the JSD
    jsd = 0.5 * kl_p_a + 0.5 * kl_m_a

    return jsd


def sym_ce(weights, distributions, target, sample_weights):
    # Doesn't check if the distributions are valid probability distributions
    # Make sure weights, each row of distributsions, and target all sum to 1.
    combined = distributions @ weights
    if sample_weights is None:
        sample_weights = 1 / distributions.shape[1]
    ce = -np.sum(target * np.log(combined) * sample_weights) - np.sum(
        combined * np.log(target) * sample_weights
    )
    return ce


def scipy_optimize(
    target_distribution, distributions, verbose=True, obj_fn=js_div, sample_weights=None
):
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

    # normalize and pad distributions for stability
    distributions /= np.sum(distributions, axis=0)
    distributions += 1e-12
    target_distribution /= np.sum(target_distribution)
    target_distribution += 1e-12

    # Solve optimization problem
    result = minimize(
        obj_fn,
        initial_weights,
        args=(distributions, target_distribution, sample_weights),
        bounds=bounds,
        constraints=constraints,
        options={"disp": verbose, "maxiter": 500},
    )

    if not result.success:
        print(f"Optimization warning: {result.message}")

    return result.x


# endregion: scipy optimization


# region: relevance functions


def h2h(ref, ests, metric="b15") -> pd.Series:
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "b30", "b05", "t", "l", "l-exp"
    # Monocasting and label decoding should be done before hand.
    # Let's do the simple case of having existing metrics to compute relevance.

    # h2h should check for boundary monotonicity. and raise error if not monotonic.
    if not ref.has_monotonic_bs():
        warnings.warn(
            "Reference must have monotonic boundaries. monocasting using 1layer label strategy...",
            stacklevel=2,
        )
        ref = ref.monocast("1layer")
    for est_key in ests.keys():
        if not ests[est_key].has_monotonic_bs():
            warnings.warn(
                "Estimate must have monotonic boundaries. monocasting using 1layer label strat...",
                stacklevel=2,
            )
            ests[est_key] = ests[est_key].monocast("1layer")

    rel = pd.Series(index=ests.keys(), dtype=float, name=metric)

    if metric[0] == "b":
        window = int(metric[1:]) * 0.1
        for key, est in ests.items():
            rel[key] = bmeasure3(ref, est.align(ref), window=window)[2]  # Take the f-score

    # For L-measure, check if we want to do hierarchy label expansion.
    elif metric[0] == "l":
        if metric == "l-exp":
            ref = ref.expand_labels()
        for key, est in ests.items():
            est = est.align(ref)
            if metric == "l-exp":
                est = est.expand_labels()
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


def h2hc(
    ref: MultiSegment | Segment,
    ests: dict[str, MultiSegment],
    metric: str = "bpc",
    obj_fn=js_div,
    ignore=(),
) -> pd.Series:
    # Similar to h2f, but opposed to begging bpc and lam per layer,
    # it takes the naive average to save compute.
    ests = filter_named_ms(ests, ignore=ignore)
    if metric == "bpc":
        # pick a sampling rate that gives me about less than 4k points, and no finer than 0.1 secs
        frame_size = max(0.1, ref.duration / 4000)
        # print(f"Using frame size: {frame_size}")
        time_grid = build_time_grid(ref, frame_size)
        y = ref.contour("prob").bpc(bw=0.8, time_grid=time_grid)
        sample_weights = np.ones_like(y)
        # Get a bpc for each est in ests.
        x = dict()
        for name, est in ests.items():
            est = est.align(ref)
            x[name] = est.contour("prob").bpc(bw=0.8, time_grid=time_grid)
        x = pd.DataFrame(x, index=time_grid)
    elif metric == "lam":
        # let's do it in the SAM space. Find the common time grid
        time_grid = common_itvls(ref.layers + combine_ms(ests).layers)
        # I need area for each sample point as well
        sample_points, sample_weights = bs2uv(time_grid, min_dur=2.0)
        # should I use depth or prob for ref lam?
        ref_lam_values = ref.lam(strategy="prob").sample(sample_points)

        est_lams = {name: est.lam("prob").sample(sample_points) for name, est in ests.items()}
        y = pd.Series(ref_lam_values, index=sample_points.T.tolist())
        x = pd.DataFrame(est_lams, index=sample_points.T.tolist())
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    # run scipy optimize
    weights = pd.Series(
        scipy_optimize(y, x, obj_fn=obj_fn, sample_weights=sample_weights),
        index=x.columns,
        name=metric,
    )

    return weights


def h2f(
    ref: MultiSegment | Segment,
    ests: dict[str, MultiSegment],
    metric: str = "bpc",
    obj_fn=js_div,
    ignore=(),
) -> pd.Series:
    # Get the relevance of each estimate with respect to the reference using metric
    # Metric can be "bpc", "lam"
    # right now the gridtime is predefined: for bpc it's 0.1 second in build_time_grid
    # for lam it's 0.5 second for lam
    ests = filter_named_ms(ests, ignore=ignore)
    est = combine_ms(ests).align(ref)

    if metric == "bpc":
        # pick a sampling rate that gives me about less than 4k points, and no finer than 0.1 secs
        frame_size = max(0.1, ref.duration / 4000)
        # print(f"Using frame size: {frame_size}")
        time_grid = build_time_grid(ref, frame_size)
        y = ref.contour("prob").bpc(bw=0.8, time_grid=time_grid)

        if len(est) == 0:
            raise ValueError("No valid estimated layers found.")
        x = est.bpcs(bw=0.8, time_grid=time_grid)
        sample_weights = np.ones_like(y)

    elif metric == "lam":
        # let's do it in the SAM space. Find the common time grid
        time_grid = common_itvls(ref.layers + est.layers)
        # I need area for each sample point as well
        sample_points, sample_weights = bs2uv(time_grid, min_dur=2.0)
        # should I use depth or prob for ref lam?
        ref_lam_values = ref.expand_labels().lam(strategy="prob").sample(sample_points)

        est_lams = {layer.name: layer.lam_pdf.sample(sample_points) for layer in est}
        y = pd.Series(ref_lam_values, index=sample_points.T.tolist())
        x = pd.DataFrame(est_lams, index=sample_points.T.tolist())

    else:
        raise ValueError(f"Metric {metric} not recognized.")

    # run scipy optimize
    weights = pd.Series(
        scipy_optimize(y, x, obj_fn=obj_fn, sample_weights=sample_weights),
        index=x.columns,
        name=metric,
    )

    return weights


def f2f(ref_layer, est, metric="hr15") -> pd.Series:
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


def rel_suite(ref, ests):
    mono_ests = {name: ms.monocast() for name, ms in ests.items()}

    rel_b15 = h2h(ref, mono_ests, metric="b15")
    rel_t = h2h(ref, mono_ests, metric="t")
    rel_l = h2h(ref, mono_ests, metric="l-exp")
    rel_bpc = h2f(ref, ests, metric="bpc")
    rel_lam = h2f(ref, ests, metric="lam")

    rel_mat = pd.concat(
        [
            aggregate_relevance(rel_bpc, name="bpc"),
            aggregate_relevance(rel_lam, name="lam"),
            rel_b15,
            rel_t,
            rel_l,
        ],
        axis=1,
    )

    return rel_mat


# endregion: relevance functions


# region: compatibility diagrams


def cd_h2h(
    refs: dict[str, MultiSegment], ests: dict[str, MultiSegment], metric="b15"
) -> pd.DataFrame:
    """
    columns:refs, rows: ests
    """
    rels = []
    for name, ref in refs.items():
        r = h2h(ref, ests, metric=metric)
        r.name = name
        rels.append(r)

    df = pd.concat(rels, axis=1)
    df.name = metric
    return df


def combo_cds_ignore(name: str) -> list[re.Pattern[str] | str]:
    if "_" not in name:
        return [name]
    parts = name.split("_")
    prefix = parts[0]
    suffix = parts[-1]
    return [re.compile(rf"^{re.escape(prefix)}_"), re.compile(rf"_{re.escape(suffix)}$")]


def cd_h2hc(
    refs: dict[str, MultiSegment], ests: dict[str, MultiSegment], metric="bpc", obj_fn=js_div
) -> pd.DataFrame:
    rels = []

    for name, ref in refs.items():
        r = h2hc(ref, ests, metric=metric, ignore=combo_cds_ignore(name), obj_fn=obj_fn)
        r.name = name
        rels.append(r)

    df = pd.concat(rels, axis=1)
    df.name = metric
    return df.reindex(ests.keys())


def cd_h2f(
    refs: dict[str, MultiSegment],
    ests: dict[str, MultiSegment],
    metric="bpc",
    agg=False,
    obj_fn=js_div,
) -> pd.DataFrame:
    rels = []
    index = []
    for name, ref in refs.items():
        r = h2f(ref, ests, metric=metric, ignore=combo_cds_ignore(name), obj_fn=obj_fn)
        if agg:
            r = aggregate_relevance(r)
            r.name = name
            rels.append(r)
            index.append(name)
        else:
            for layer in combine_ms({name: ref}).layers:
                r.name = layer.name
                rels.append(r.copy())
                index.append(layer.name)
    df = pd.concat(rels, axis=1).reindex(index)
    df.name = metric
    return df


def cd_f2f(
    ref: MultiSegment | dict[str, MultiSegment],
    est: MultiSegment | dict[str, MultiSegment],
    metric="hr15",
) -> pd.DataFrame:
    rels = []
    if isinstance(ref, dict):
        ref = combine_ms(ref)
    if isinstance(est, dict):
        est = combine_ms(est)
    for ref_layer in ref:
        r = f2f(ref_layer, est, metric=metric)
        r.name = ref_layer.name
        rels.append(r)

    df = pd.concat(rels, axis=1)
    df.name = metric
    return df


def aggregate_relevance(weights: pd.Series, delimiter=":", name=None) -> pd.Series:
    # aggregate layers relevance for a hierarchy if aggregate_hierarchy is True
    agg_weights = dict()
    hier_names = weights.index.str.split(delimiter).str[0].unique()
    for hier_name in hier_names:
        hier_weights = weights.filter(regex=f"{hier_name}{delimiter}")
        agg_weights[hier_name] = hier_weights.sum(skipna=True)
    return pd.Series(agg_weights, index=hier_names, name=name)


def cd_suite(ests: dict[str, MultiSegment]) -> dict[str, pd.DataFrame]:
    mono_ests = {name: ms.monocast() for name, ms in ests.items()}
    cds = dict()
    cds["b05"] = cd_h2h(mono_ests, mono_ests, metric="b05")
    cds["b15"] = cd_h2h(mono_ests, mono_ests, metric="b15")
    cds["b30"] = cd_h2h(mono_ests, mono_ests, metric="b30")
    cds["lex"] = cd_h2h(mono_ests, mono_ests, metric="l-exp")
    cds["bpc"] = cd_h2f(ests, ests, metric="bpc", agg=False)
    cds["lam"] = cd_h2f(ests, ests, metric="lam", agg=False)
    return cds


# endregion: compatibility diagrams

# region: weight estimation from CD


@dataclass
class CompDiagramStats:
    """Structure for compatibility diagram statistics."""

    w: pd.Series  # normalized weights
    wentropy: float  # entropy of weights
    evmax: float  # largest eigenvalue
    evgap: float  # first eigen gap
    evals: pd.Series  # eigenvalues

    def to_dict(self):
        return self.__dict__


# I want to use networkX objects now...
def cd2w(cd: pd.DataFrame, pad=0.001) -> CompDiagramStats:
    """
    cd: compatibility diagram
    returns: dict of computed weights and other statistics
    """
    cd = cd.fillna(0)

    if pad > 0:
        cd = cd * (1 - pad) + pad
    # Decompose and sort descending
    evals, evecs = np.linalg.eig(cd.values)
    idx = np.abs(evals).argsort()[::-1]
    evals = np.abs(evals[idx])
    evecs = evecs[:, idx]

    # first eigen Vector
    evec_1 = np.abs(evecs[:, 0])
    w = pd.Series(evec_1 / np.sum(evec_1), index=cd.index)

    # Normalize to sum to 1
    return CompDiagramStats(
        w=w,
        wentropy=stats.entropy(w),
        evmax=evals[0] / np.sum(evals),
        evgap=evals[0] / evals[1],
        evals=pd.Series(evals),
    )


def cd2nx(cd: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    import networkx as nx

    # Check if cd is symmetric
    if cd.equals(cd.T):
        G = nx.from_pandas_adjacency(cd.fillna(0), create_using=nx.Graph())
        centrality = pd.Series(nx.eigenvector_centrality(G, weight="weight", max_iter=1000))
        spectrum = pd.Series(np.sort(np.abs(nx.adjacency_spectrum(G)))[::-1])
    else:
        G = nx.from_pandas_adjacency(cd.T.fillna(0), create_using=nx.DiGraph())
        centrality = pd.Series(nx.pagerank(G, alpha=0.99, max_iter=1000))
        spectrum = pd.Series(np.sort(np.abs(nx.adjacency_spectrum(G)))[::-1])
    return centrality, spectrum
