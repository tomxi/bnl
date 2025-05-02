import numpy as np
from mir_eval.util import boundaries_to_intervals
from .core import H, S, levels2H
from .external import reindex
from .utils import best_matching_label
from .metrics import vmeasure


def prune_identical_levels(hier, boundary_only=False):
    """Prune identical levels."""
    if boundary_only:
        hier = H(hier.itvls, sr=hier.sr, Bhat_bw=hier.Bhat_bw)

    new_levels = [hier.levels[0]]
    for i in range(1, hier.d):
        if np.array_equal(hier.levels[i].beta, hier.levels[i - 1].beta):
            common_bs = hier.levels[i].beta
            if np.allclose(
                hier.levels[i].A(bs=common_bs),
                new_levels[-1].A(bs=common_bs),
            ):
                continue
        new_levels.append(hier.levels[i])
    return levels2H(new_levels, sr=hier.sr, Bhat_bw=hier.Bhat_bw)


def squash_levels(hier, boundary_only=False, max_depth=3, remove_single_itvls=True):
    """Squash levels. by removing the level that adds the least information according to vmeasure."""
    if boundary_only:
        hier = H(hier.itvls, sr=hier.sr, Bhat_bw=hier.Bhat_bw)

    new_levels = hier.levels.copy()
    if remove_single_itvls:
        for lvl in new_levels:
            # remove single interval levels
            if len(lvl.itvls) == 1:
                new_levels.remove(lvl)
                continue

    # max_depth = None means no limit
    while max_depth is not None and len(new_levels) > max_depth:
        # get rid of the level that adds the least information
        # look at vmeasure between all consecutive levels,
        # get the one with the lowest vmeasure with the next level
        level_pairs = [
            (new_levels[i], new_levels[i + 1]) for i in range(len(new_levels) - 1)
        ]
        v_f1 = [
            vmeasure(lv1.itvls, lv1.labels, lv2.itvls, lv2.labels)[2]
            for lv1, lv2 in level_pairs
        ]
        new_levels.pop(np.argmax(v_f1))
    return levels2H(new_levels, sr=hier.sr, Bhat_bw=hier.Bhat_bw)


def relabel(hier, strategy="max_overlap"):
    """Re-label the hierarchical segmentation."""
    if strategy == "max_overlap":
        ext_format = [[i, l] for i, l in zip(hier.itvls, hier.labels)]
        new_hier = reindex(ext_format)
        new_labels = [lvl[1] for lvl in new_hier]

    elif strategy == "unique":
        new_labels = None

    return H(hier.itvls, new_labels, sr=hier.sr, Bhat_bw=hier.Bhat_bw)


def has_mono_L(hier):
    """Check if labels are monotonic across levels."""
    return np.allclose(hier.A(bs=hier.beta), hier.Astar(bs=hier.beta))


def has_mono_B(H):
    """Check if boundaries are monotonic across levels."""
    return all(
        set(H.levels[i - 1].beta).issubset(H.levels[i].beta) for i in range(1, H.d)
    )


def force_mono_B(hier, absorb_window=0):
    """Force monotonic boundaries across levels."""
    new_levels = []
    for level in hier.levels:
        if len(new_levels) == 0:
            new_levels.append(level)
            continue
        parent_bounds = new_levels[-1].beta
        child_bounds = level.beta
        if set(parent_bounds).issubset(child_bounds):
            new_levels.append(level)
        else:
            if absorb_window:
                child_bounds = set(
                    b
                    for b in child_bounds
                    if all(abs(b - pb) > absorb_window for pb in parent_bounds)
                )
            new_child_bounds = sorted(list(set(parent_bounds).union(child_bounds)))
            new_child_itvls = boundaries_to_intervals(new_child_bounds)
            new_child_labels = [
                best_matching_label(query_itvl, level.itvls, level.labels)
                for query_itvl in new_child_itvls
            ]
            new_levels.append(S(new_child_itvls, new_child_labels))
    return levels2H(new_levels, sr=hier.sr, Bhat_bw=hier.Bhat_bw)


def force_mono_L(hier, absorb_window=0):
    """Force monotonic labels across levels."""
    self_mono_B = force_mono_B(hier, absorb_window=absorb_window)
    if has_mono_L(self_mono_B):
        return self_mono_B
    new_levels = [self_mono_B.levels[0]]
    for i in range(1, self_mono_B.d):
        level = self_mono_B.levels[i]
        parent_level = new_levels[-1]
        new_child_labels = [parent_level(b) + "." + level(b) for b in level.beta[:-1]]
        new_levels.append(S(level.itvls, new_child_labels))
    return levels2H(new_levels, sr=hier.sr, Bhat_bw=hier.Bhat_bw)
