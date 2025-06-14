import numpy as np
from scipy import stats
import librosa, warnings
from matplotlib import pyplot as plt
from mir_eval.util import intervals_to_boundaries, boundaries_to_intervals
import matplotlib.patches as mpatches

from .formatting import (
    mireval2multi,
    multi2mireval,
    mirevalflat2openseg,
    openseg2mirevalflat,
)
from .external import expand_hierarchy
from . import viz, utils

__all__ = ["S", "H", "multi2H", "levels2H", "flat2S", "sal2H"]


class S:
    """A flat segmentation, labeled intervals."""

    def __init__(self, itvls, labels=None, sr=None, Bhat_bw=None, time_decimal=4):
        """Initialize the flat segmentation."""

        if labels is None:
            labels = list(range(len(itvls)))
        self.labels = labels

        # Build Lstar and T
        self.Lstar = {round(b, time_decimal): l for (b, e), l in zip(itvls, labels)}
        self.T = round(itvls[-1][-1], time_decimal)
        self.beta = np.array(sorted(set(self.Lstar.keys()).union([self.T])))
        self.seg_dur = self.beta[1:] - self.beta[:-1]
        self.T0 = self.beta[0]
        self.itvls = boundaries_to_intervals(self.beta)
        self.anno = mirevalflat2openseg(self.itvls, self.labels)

        # Build BSC and ticks
        # Lazy init these attributes:
        self._Bhat = None
        self.Bhat_bw = None
        self.sr = None
        self.ticks = None
        if sr:
            self.update_sr(sr)
        if Bhat_bw is not None:
            self.update_bw(Bhat_bw)

    def update_bw(self, bw):
        """Update bandwidth for Bhat calculation.
        Populates ._Bhat and .Bhat_bw
        """
        if self.Bhat_bw == bw:
            return None
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
        return None

    def update_sr(self, sr):
        """Update sampling rate and ticks."""
        if self.sr == sr:
            return None
        self.sr = float(sr)
        ## Use the same logic as mir_eval to build the ticks
        # Figure out how many frames we need
        from mir_eval.hierarchy import _round

        frame_size = 1.0 / self.sr
        n_frames = int(
            (_round(self.T, frame_size) - _round(self.T0, frame_size)) / frame_size
        )

        self.ticks = np.arange(n_frames + 1) * frame_size + self.T0
        return None

    def B(self, x):
        """Return whether x is a boundary with integers 1/0"""
        return int(x in self.beta or (x == self.T0) or (x == self.T))

    def Bhat(self, ts=None):
        """Return the boundary salience curve at given time steps."""
        if ts is None:
            ts = self.ticks
        ts = np.array(ts)
        if self.Bhat_bw is None:
            warnings.warn("Bhat_bw is not set. setting it to 1.")
            self.update_bw(1)

        return self._Bhat(ts)

    def A(self, bs=None, compare_fn=np.equal):
        """Return the label agreement indicator for given boundaries.
        when substituting compare_fn to np.greater, it can be used to get
        significant pairs when the labels are comparable.
        compare_fn needs to support .outer
        """
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs)))
        ts = (bs[1:] + bs[:-1]) / 2  # Sample label from mid-points of each frame
        labels_arr = np.array([self(t) for t in ts])
        return compare_fn.outer(labels_arr, labels_arr).astype(int)

    def Ahat(self, bs=None, compare_fn=np.equal):
        """Return the label agreement matrix.
        it's the indicator normalized by the area of the segments duration square.
        """
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs)))
        lai = self.A(bs=bs, compare_fn=compare_fn)
        seg_dur = bs[1:] - bs[:-1]
        seg_dur_area_mat = np.outer(seg_dur, seg_dur)
        total_area = np.sum(seg_dur_area_mat * lai)
        return lai / total_area if total_area > 0 else lai

    def plot(self, ax=None, **kwargs):
        """Plot the segmentation. using viz.segment()"""
        # Smallest segmnet to total duration ratio
        # Get the smallest 3 segment durations

        # find a good default width for the plot
        average_seg_dur = np.mean(self.seg_dur)
        atom_size = (average_seg_dur + np.min(self.seg_dur)) / 2.0
        # Get the number of atoms in the segmentation
        num_atoms = np.sum(self.seg_dur) / atom_size
        default_width = min(max(3.5, num_atoms * 0.35), 12)

        # Kwargs handling
        new_kwargs = dict(
            text=True, ytick="", time_ticks=True, figsize=(default_width, 0.5)
        )
        new_kwargs.update(kwargs)
        if ax is None:
            fig = plt.figure(figsize=new_kwargs["figsize"])
            ax = fig.add_subplot(111)

        new_kwargs.pop("figsize")

        # Plot the segments
        return viz.segment(self.itvls, self.labels, ax=ax, **new_kwargs)

    def expand(self, format="slm", always_include=False):
        """Expand the segmentation into a hierarchical format."""
        # Convert to hierarchical format using the specified format
        expanded_levels = [
            flat2S(l)
            for l in expand_hierarchy(
                self.anno, dataset=format, always_include=always_include
            )
        ]
        return levels2H(expanded_levels, sr=self.sr, Bhat_bw=self.Bhat_bw)

    def __call__(self, x):
        if not (self.T0 <= x <= self.T):
            raise IndexError(
                f"RANGE: {x} outside the range of this segmentation {self.T0, self.T}!"
            )
        elif x == self.T:
            return self.Lstar[self.beta[-2]]

        idx = np.searchsorted(self.beta, x, side="right") - 1
        return self.Lstar[self.beta[idx]]


class H:
    """A hierarchical segmentation composed of multiple flat segmentations."""

    def __init__(self, itvls, labels=None, sr=None, Bhat_bw=None, time_decimal=4):
        """Initialize the hierarchical segmentation."""
        # Validate same start/end points across levels
        start_points = [round(level[0][0], time_decimal) for level in itvls]
        end_points = [round(level[-1][-1], time_decimal) for level in itvls]

        if len(set(start_points)) != 1 or len(set(end_points)) != 1:
            # Make all level start/end points the same
            start_point = min(start_points)
            end_point = max(end_points)
            for level in itvls:
                level[0][0] = start_point
                level[-1][-1] = end_point

        if labels is None:
            labels = [[str(s) for s in itvl[:, 0]] for itvl in itvls]

        self.levels = [
            S(i, l, sr=sr, Bhat_bw=Bhat_bw, time_decimal=time_decimal)
            for i, l in zip(itvls, labels)
        ]
        self.itvls = [l.itvls for l in self.levels]
        self.labels = [l.labels for l in self.levels]
        self.anno = mireval2multi(self.itvls, self.labels)
        self.d = len(self.levels)
        self.T0, self.T = self.levels[0].T0, self.levels[0].T
        self.beta = np.unique(np.concatenate([seg.beta for seg in self.levels]))
        self.update_bw(Bhat_bw)
        self.update_sr(sr)

    def update_sr(self, sr):
        """Update sampling rate and ticks for all levels."""
        for level in self.levels:
            level.update_sr(sr)
        self.sr = level.sr
        self.ticks = level.ticks

    def update_bw(self, Bhat_bw):
        for lvl in self.levels:
            lvl.update_bw(Bhat_bw)
        self.Bhat_bw = Bhat_bw

    def Ahats(self, bs=None):
        """Return the normalized label agreement matrices for all levels."""
        return np.asarray([lvl.Ahat(bs) for lvl in self.levels])

    def Bhats(self, ts=None):
        """Return the smoothed boundary strengths for all levels."""
        if self.Bhat_bw is None:
            warnings.warn("Bhat_bw is not set. setting it to 1.")
            self.update_bw(1)
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

    def A(self, bs=None, compare_fn=np.equal):
        """Return the sum of label agreement mats for all levels
        with segments defined by boundaires bs.
        """
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(bs))
        return sum(level.A(bs=bs, compare_fn=compare_fn) for level in self.levels)

    def B(self, t):
        """Return the boundary salience by finding the coarsest level that a boundary appears in."""
        for k in range(self.d):
            if self.levels[k].B(t):
                return self.d - k
        return 0

    def Astar(self, bs=None):
        """Return the deepest level where labels are identical.
        It's the Annotation Meet Matrix as its defined in prior literature.
        """
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs)))
        Ahats = self.Ahats(bs=bs)
        indexed_Ahats = np.array(
            [(level + 1) * (Ahats[level] > 0).astype(int) for level in range(self.d)]
        )
        return np.max(indexed_Ahats, axis=0)

    def M(self, bs=None, level_weights=None):
        """Return the resampled agreement area matrix."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs)))
        all_bs = np.array(sorted(set(self.beta).union(bs)))
        seg_dur = all_bs[1:] - all_bs[:-1]
        seg_agreement_area = np.outer(seg_dur, seg_dur)
        return utils.resample_matrix(
            seg_agreement_area * self.Ahat(bs=all_bs, weights=level_weights), all_bs, bs
        )

    def Mhat(self, bs=None, level_weights=None):
        """Return the normalized resampled agreement area matrix."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs)))
        seg_dur = bs[1:] - bs[:-1]
        return self.M(bs, level_weights=level_weights) / np.outer(seg_dur, seg_dur)

    def plot(
        self,
        axs=None,
        text=True,
        legend=False,  #
        legend_offset=0.2,
        time_ticks=True,
        **create_fig_kwargs,
    ):
        """Plot the hierarchical segmentation."""
        # kwargs handling
        fig_kw = dict(
            figsize=(5, 0.4 * self.d + 0.5),
            h_ratios=[1] * self.d,
            w_ratios=[1],
        )
        fig_kw.update(create_fig_kwargs)

        if axs is None:
            _, axs = viz.create_fig(**fig_kw)
            # flatten nested list of axes
            axs = np.array(axs).flatten()

        # Check len(axs) and self.d is the same
        if len(axs) < self.d:
            raise ValueError(
                f"Number of axes ({len(axs)}) is smaller than number of levels ({self.d})."
            )

        ## Starting to do real work...
        # build stylemap from labels
        style_map = viz.label_style_dict(labels=self.labels)

        # Plot each level
        if self.d > 1:
            for i in range(self.d - 1):
                self.levels[i].plot(
                    ax=axs[i],
                    ytick=i + 1,
                    time_ticks=False,
                    text=text,
                    style_map=style_map,
                )
        self.levels[-1].plot(
            ax=axs[-1],
            ytick=self.d,
            time_ticks=time_ticks,
            text=text,
            style_map=style_map,
        )
        # Set legend
        if legend:
            legend_handles = [mpatches.Patch(**style) for style in style_map.values()]
            axs[-1].legend(
                handles=legend_handles,
                labels=list(style_map.keys()),
                loc="upper center",
                fontsize="small",
                ncol=legend,
                bbox_to_anchor=(0.5, -legend_offset),
            )

        return axs[0].get_figure(), axs

    def decode_B(
        self,
        depth=None,
        pre_max=0.8,
        post_max=0.8,
        pre_avg=0.3,
        post_avg=0.3,
        delta=1e-3,
        wait=1,
        level_weights=None,
        sr=None,
        bw=None,
    ):
        """Return a hierarchical segmentation with monotonic boundaries, with the specified number of levels."""
        # decode_b needs sr and bw.
        if sr is not None:
            self.update_sr(sr)
        if bw is not None:
            self.update_bw(bw)
        if self.sr is None or self.Bhat_bw is None:
            raise ValueError("sr and bw must be set for decode_B")

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
        if depth is None:
            depth = self.d
        intervals = utils.cluster_boundaries(boundaries, novelty, self.ticks, depth)

        return intervals

    def decode_L(self, itvls, min_k=2):
        """decode labels from the coarsest to most fine, using increasing k from eigen-gap."""
        current_k = min_k
        labs = []
        for lvl_itvls in itvls:
            bs = intervals_to_boundaries(lvl_itvls)
            M = self.Mhat(bs=bs)
            lab, current_k = utils.eigen_gap_scluster(M, min_k=current_k)
            labs.append(lab)
        return H(itvls, labs, sr=self.sr, Bhat_bw=self.Bhat_bw)

    def decode(self, depth=4, min_k=2, **kwargs):
        """Decode the hierarchical segmentation."""
        # Decode boundaries
        new_itvls = self.decode_B(depth=depth, **kwargs)
        # Decode labels
        return self.decode_L(new_itvls, min_k=min_k)

    def expand(self, format="slm", always_include=False):
        """Expand the hierarchy annotations using the specified format.

        Args:
            format (str): Dataset format to use for expansion (default: 'slm')
            always_include (bool): Whether to always include original annotations

        Returns:
            H: A hierarchical segmentation with all expanded levels
        """
        expanded_levels = []
        for level in self.levels:
            # Expand each level and collect all resulting levels
            expanded = level.expand(format=format, always_include=always_include)
            expanded_levels.extend(expanded.levels)

        return levels2H(expanded_levels, sr=self.sr, Bhat_bw=self.Bhat_bw)

    def __call__(self, x):
        return [l(x) for l in self.levels]


def levels2H(levels, sr=None, Bhat_bw=None):
    """Convert a list of levels (S) to a hierarchical format."""
    itvls = [l.itvls for l in levels]
    lbls = [l.labels for l in levels]
    return H(itvls, lbls, sr=sr, Bhat_bw=Bhat_bw)


def multi2H(anno, sr=None, Bhat_bw=None):
    """Convert multiple segments to hierarchical format."""
    segments = multi2mireval(anno)
    return H(*segments, sr=sr, Bhat_bw=Bhat_bw)


def flat2S(anno, sr=None, Bhat_bw=None):
    """Convert flat annotations to hierarchical format."""
    segment = openseg2mirevalflat(anno)
    return S(*segment, sr=sr, Bhat_bw=Bhat_bw)


def sal2H(bs_sal, sr=None, Bhat_bw=None):
    """Convert salience dictionary to hierarchical format."""
    boundaries_by_level = []
    total_levels = max(bs_sal.values())
    for level in range(total_levels, 0, -1):  # Iterate from most to least salient
        level_boundaries = [b for b in bs_sal if bs_sal[b] >= level]
        boundaries_by_level.append(boundaries_to_intervals(sorted(level_boundaries)))
    return H(boundaries_by_level, sr=sr, Bhat_bw=Bhat_bw).prune_identical_levels()
