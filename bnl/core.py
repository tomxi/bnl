import numpy as np
from scipy import stats
import librosa, mir_eval

from .utils import eigen_gap_scluster, resample_matrix, cluster_boundaries
from .viz import multi_seg
from .formatting import mireval2multi, multi2mireval
from mir_eval.util import intervals_to_boundaries, boundaries_to_intervals


class S:
    """A flat segmentation, labeled intervals."""

    def __init__(self, itvls, labels=None, sr=10, Bhat_bw=1, time_decimal=3):
        """Initialize the flat segmentation."""
        self.itvls = itvls
        if labels is None:
            labels = [str(itv) for itv in itvls]
        self.labels = labels
        self.anno = mireval2multi([itvls], [labels])

        # Build Lstar and T
        self.Lstar = {
            round(b, time_decimal): str(l) for (b, e), l in zip(itvls, labels)
        }
        self.beta = np.array(sorted(set(self.Lstar.keys()).union([itvls[-1][-1]])))
        self.T0, self.T = self.beta[0], self.beta[-1]

        # Build BSC and ticks
        self.update_sr(sr)
        self.update_bw(Bhat_bw)

        self.seg_dur = self.beta[1:] - self.beta[:-1]
        self.seg_dur_area_mat = np.outer(self.seg_dur, self.seg_dur)
        self.total_label_agreement_area = np.sum(
            self.seg_dur_area_mat * self.A(bs=self.beta)
        )

    def update_bw(self, bw):
        """Update bandwidth for Bhat calculation."""
        if hasattr(self, "Bhat_bw") and self.Bhat_bw == bw:
            return
        self.Bhat_bw = bw
        boundaries = self.beta[1:-1]
        if len(boundaries) == 0:
            self._Bhat = lambda ts: np.array([0 for _ in ts])
        elif len(boundaries) == 1:
            self._Bhat = stats.norm(loc=boundaries[0], scale=bw).pdf
        else:
            kde_bw = bw / boundaries.std(ddof=1) if boundaries.std(ddof=1) != 0 else bw
            kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
            self._Bhat = kde

    def update_sr(self, sr):
        """Update sampling rate and ticks."""
        if hasattr(self, "sr") and self.sr == sr:
            return
        self.sr = sr
        self.ticks = np.linspace(
            self.T0, self.T, int(np.round((self.T - self.T0) * self.sr)) + 1
        )

    def L(self, x):
        """Return the label for a given time x."""
        if not (self.T0 <= x <= self.T):
            raise IndexError(f"RANGE: {x} outside the range of this segmentation!")
        idx = np.searchsorted(self.beta, x, side="right") - 1
        return self.Lstar[self.beta[idx]]

    def B(self, x):
        """Return whether x is a boundary with integers 1/0"""
        return int(x in self.beta or (x == self.T0) or (x == self.T))

    def Bhat(self, ts=None):
        """Return the boundary salience curve at given time steps."""
        if ts is None:
            ts = self.ticks
        ts = np.array(ts)
        return self._Bhat(ts)

    def A(self, bs=None):
        """Return the label agreement indicator for given boundaries."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        ts = (bs[1:] + bs[:-1]) / 2  # Sample label from mid-points of each frame
        sampled_anno = self.anno.to_samples(ts)
        sample_labels = [obs[0]["label"] for obs in sampled_anno]
        return np.equal.outer(sample_labels, sample_labels).astype(float)

    def Ahat(self, bs=None):
        """Return the label agreement matrix."""
        return self.A(bs) / self.total_label_agreement_area

    def plot(self, **kwargs):
        """Plot the segmentation."""
        new_kwargs = dict(text=True, legend_ncol=0, figsize=(6, 0.9), y_label=False)
        new_kwargs.update(kwargs)
        return multi_seg(self.anno, **new_kwargs)


class H:
    """A hierarchical segmentation composed of multiple flat segmentations."""

    def __init__(self, itvls, labels=None, sr=10, Bhat_bw=1, time_decimal=3):
        """Initialize the hierarchical segmentation."""
        # Validate same start/end points across levels
        start_points = [level[0][0] for level in itvls]
        end_points = [level[-1][-1] for level in itvls]

        if len(set(start_points)) != 1 or len(set(end_points)) != 1:
            raise ValueError(
                f"All levels must share same start/end points. Got starts {start_points} and ends {end_points}."
            )

        if labels is None:
            labels = itvls
        self.itvls = itvls
        self.labels = labels
        self.anno = mireval2multi(itvls, labels)
        self.levels = [
            S(i, l, sr=sr, Bhat_bw=Bhat_bw, time_decimal=time_decimal)
            for i, l in zip(itvls, labels)
        ]
        self.d = len(self.levels)
        self.T0, self.T = self.levels[0].T0, self.levels[0].T
        self.beta = np.unique(np.concatenate([seg.beta for seg in self.levels]))
        self.update_bw(Bhat_bw)
        self.update_sr(sr)

    def update_sr(self, sr):
        """Update sampling rate and ticks."""
        if hasattr(self, "sr") and self.sr == sr:
            return
        self.sr = sr
        self.ticks = np.linspace(
            self.T0, self.T, int(np.round((self.T - self.T0) * self.sr)) + 1
        )

    def update_bw(self, Bhat_bw):
        if hasattr(self, "Bhat_bw") and self.Bhat_bw == Bhat_bw:
            return
        self.Bhat_bw = Bhat_bw
        for lvl in self.levels:
            lvl.update_bw(Bhat_bw)

    def Ahats(self, bs=None):
        """Return the normalized label agreement matrices for all levels."""
        return np.asarray([lvl.Ahat(bs) for lvl in self.levels])

    def Bhats(self, ts=None):
        """Return the smoothed boundary strengths for all levels."""
        return np.asarray([lvl.Bhat(ts) for lvl in self.levels])

    def Ahat(self, bs=None, weights=None):
        """Return the weighted normalized label agreement matrix."""
        if weights is None:
            weights = np.ones(self.d)
        weights /= np.sum(weights)
        weighted = np.array(weights).reshape(-1, 1, 1) * self.Ahats(bs)
        return np.sum(weighted, axis=0)

    def Bhat(self, ts=None, weights=None):
        """Return the weighted smoothed boundary strength."""
        if weights is None:
            weights = np.ones(self.d)
        weights /= np.sum(weights)
        weighted = np.array(weights).reshape(-1, 1) * self.Bhats(ts)
        return np.sum(weighted, axis=0)

    def A(self, bs=None):
        """Return the sum of label agreement mats for all levels
        with segments defined by boundaires bs.
        """
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        return sum(level.A(bs=bs) for level in self.levels)

    def B(self):
        """Return the boundary count across all levels."""
        rated_boundaries = dict()
        for b in self.beta:
            rated_boundaries[b] = sum(seg.B(b) for seg in self.levels)
        return rated_boundaries

    def Astar(self, bs=None):
        """Return the deepest level where labels are identical.
        It's the Annotation Meet Matrix as its defined in prior literature.
        """
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        Ahats = self.Ahats(bs=bs)
        indexed_Ahats = np.array(
            [(level + 1) * (Ahats[level] > 0).astype(int) for level in range(self.d)]
        )
        return np.max(indexed_Ahats, axis=0)

    def has_monoA(self):
        """Check if labels are monotonic across levels."""
        return np.allclose(self.A(bs=self.beta), self.Astar(bs=self.beta))

    def has_monoB(self):
        """Check if boundaries are monotonic across levels."""
        return all(
            set(self.levels[i - 1].beta).issubset(self.levels[i].beta)
            for i in range(1, self.d)
        )

    def M(self, bs=None, level_weights=None):
        """Return the resampled agreement area matrix."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        all_bs = np.array(sorted(set(self.beta).union(bs)))
        seg_dur = all_bs[1:] - all_bs[:-1]
        seg_agreement_area = np.outer(seg_dur, seg_dur)
        return resample_matrix(
            seg_agreement_area * self.Ahat(bs=all_bs, weights=level_weights), all_bs, bs
        )

    def Mhat(self, bs=None, level_weights=None):
        """Return the normalized resampled agreement area matrix."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        seg_dur = bs[1:] - bs[:-1]
        return self.M(bs, level_weights=level_weights) / np.outer(seg_dur, seg_dur)

    def plot(self, **kwargs):
        """Plot the hierarchical segmentation."""
        new_kwargs = dict()
        new_kwargs.update(kwargs)
        return multi_seg(self.anno, **new_kwargs)

    def decode_B(
        self,
        depth=4,
        pre_max=0.8,
        post_max=0.8,
        pre_avg=0.3,
        post_avg=0.3,
        delta=1e-3,
        wait=1,
        level_weights=None,
    ):
        """Return a hierarchical segmentation with monotonic boundaries, with the specified number of levels."""
        # Normalize the novelty curve
        novelty = self.Bhat(ts=self.ticks, weights=level_weights)
        novelty /= novelty.max() + 1e-10
        novelty[0], novelty[-1] = novelty.max(), novelty.max()

        # Identify boundaries using peak picking
        boundaries = librosa.util.peak_pick(
            novelty,
            pre_max=int(pre_max * self.sr),
            post_max=int(post_max * self.sr),
            pre_avg=int(pre_avg * self.sr),
            post_avg=int(post_avg * self.sr),
            delta=delta,
            wait=int(wait * self.sr),
        )

        # Ensure the first and last frames are included in the boundaries list
        boundaries = np.unique(np.concatenate(([0, len(novelty) - 1], boundaries)))

        # Convert boundaries to hierarchical intervals
        intervals = cluster_boundaries(boundaries, novelty, self.ticks, depth)

        return H(intervals, sr=self.sr, Bhat_bw=self.Bhat_bw)

    def decode_L(self, itvls, min_k=2):
        """decode labels from the coarsest to most fine, using increasing k from eigen-gap."""
        current_k = min_k
        labs = []
        for lvl_itvls in itvls:
            bs = intervals_to_boundaries(lvl_itvls)
            M = self.Mhat(bs=bs)
            lab, current_k = eigen_gap_scluster(M, min_k=current_k)
            labs.append(lab)
        return H(itvls, labs, sr=self.sr, Bhat_bw=self.Bhat_bw)

    def decode(self, depth=4, min_k=2, **kwargs):
        """Decode the hierarchical segmentation."""
        # Decode boundaries
        # print('Bhat bw:', self.Bhat_bw)
        new_H = self.decode_B(depth=depth, **kwargs)
        # Decode labels
        return self.decode_L(new_H.itvls, min_k=min_k)

    def force_mono_B(self):
        """Force monotonic boundaries across levels.
        If a boundary is present in a parent level, it has to appear in a child level.
        """
        ## Start from the first level and work down
        new_levels = []
        for level in self.levels:
            if len(new_levels) == 0:
                new_levels.append(level)
                continue
            parent_boundaries = new_levels[-1].beta
            child_boundaries = level.beta
            new_child_boundaries = sorted(
                list(set(parent_boundaries).union(child_boundaries))
            )
            new_child_labels = [level.L(b) for b in new_child_boundaries[:-1]]
            new_child_itvls = boundaries_to_intervals(new_child_boundaries)
            new_levels.append(S(new_child_itvls, new_child_labels))
        return levels2H(new_levels, sr=self.sr, Bhat_bw=self.Bhat_bw)


def levels2H(levels, sr=10, Bhat_bw=1):
    itvls = [l.itvls for l in levels]
    lbls = [l.labels for l in levels]
    return H(itvls, lbls, sr=sr, Bhat_bw=Bhat_bw)


def multi2H(anno, sr=10, Bhat_bw=1):
    return H(*multi2mireval(anno), sr=sr, Bhat_bw=Bhat_bw)
