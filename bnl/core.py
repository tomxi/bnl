import numpy as np
from scipy import stats
import librosa, warnings
from matplotlib import pyplot as plt
from mir_eval.util import intervals_to_boundaries, boundaries_to_intervals
import matplotlib.patches as mpatches

from .formats import (
    mireval2multi,
    multi2mireval,
    mirevalflat2openseg,
    openseg2mirevalflat,
)
# from .hierarchy import expand_hierarchy # Moved to Segmentation.expand to break circular import
from . import plotting as viz
from . import utils

__all__ = ["Segmentation", "levels_to_segmentation", "multi_to_segmentation", "flat_to_segmentation", "sal_to_segmentation"]

# TODO: Future Extensibility & Documentation
# 1. Registration Mechanisms:
#    - Consider adding class methods to Segmentation for registering custom handlers:
#      - e.g., `Segmentation.register_format_handler(format_name, read_func, write_func)`
#      - e.g., `Segmentation.register_metric(metric_name, metric_func)`
#      - e.g., `Segmentation.register_plotter(plot_name, plot_func)`
#    - This would allow users to extend functionality (I/O, analysis, visualization)
#      without modifying the core library, promoting a plugin-style architecture.
#
# 2. Comprehensive Documentation & Examples:
#    - All classes, methods, and functions should have thorough docstrings (e.g., NumPy/SciPy style).
#    - A gallery of examples demonstrating common use cases for the Segmentation class
#      and its interaction with formats, hierarchy, metrics, and plotting modules
#      would be highly beneficial for users. (e.g., loading data, performing analysis, plotting results).

class Segmentation:
    """
    A class to represent both flat and hierarchical segmentations.

    This class consolidates the functionalities previously found in S (flat) and H (hierarchical)
    classes. It aims to provide a unified interface for working with music segmentations.
    """
    # TODO: Review method names for clarity (e.g., B -> boundary_presence, Ahat -> label_agreement_matrix).
    #       This is a larger change and should be considered carefully for API stability.

    def __init__(self, itvls, labels=None, sr=None, Bhat_bw=None, time_decimal=4, is_hierarchical=None):
        """Initialize the segmentation.

        Args:
            itvls (list or np.ndarray): Intervals for the segmentation.
                If hierarchical, this should be a list of interval arrays (one per level).
                If flat, this should be a single interval array.
            labels (list or np.ndarray, optional): Labels for the segmentation.
                Structure should match `itvls`. Defaults to None (range of len(itvls)).
            sr (float, optional): Sampling rate. Defaults to None.
            Bhat_bw (float, optional): Bandwidth for Bhat calculation. Defaults to None.
            time_decimal (int, optional): Number of decimal places for time values. Defaults to 4.
            is_hierarchical (bool, optional): Explicitly set if the segmentation is hierarchical.
                                            If None, it's inferred from the structure of itvls.
        """
        if is_hierarchical is None:
            self.is_hierarchical = isinstance(itvls, list) and (len(itvls) > 0 and isinstance(itvls[0], (list, np.ndarray)))
        else:
            self.is_hierarchical = is_hierarchical

        if self.is_hierarchical:
            # Validate same start/end points across levels for hierarchical segmentation
            start_points = [round(level[0][0], time_decimal) for level in itvls]
            end_points = [round(level[-1][-1], time_decimal) for level in itvls]

            if len(set(start_points)) != 1 or len(set(end_points)) != 1:
                start_point = min(start_points)
                end_point = max(end_points)
                for level_itvls in itvls: # Renamed 'level' to 'level_itvls' to avoid conflict
                    level_itvls[0][0] = start_point
                    level_itvls[-1][-1] = end_point

            if labels is None:
                labels = [[str(s) for s in lvl_itvl[:, 0]] for lvl_itvl in itvls]

            self.levels = [
                Segmentation(i, l, sr=sr, Bhat_bw=Bhat_bw, time_decimal=time_decimal, is_hierarchical=False)
                for i, l in zip(itvls, labels)
            ]
            self.itvls = [l.itvls for l in self.levels]
            self.labels = [l.labels for l in self.levels]
            self.anno = mireval2multi(self.itvls, self.labels)
            self.d = len(self.levels)
            self.T0, self.T = self.levels[0].T0, self.levels[0].T
            self.beta = np.unique(np.concatenate([seg.beta for seg in self.levels]))
        else: # Flat segmentation
            if labels is None:
                labels = list(range(len(itvls)))
            self.labels = labels
            self.Lstar = {round(b, time_decimal): l for (b, e), l in zip(itvls, labels)}
            self.T = round(itvls[-1][-1], time_decimal)
            self.beta = np.array(sorted(set(self.Lstar.keys()).union([self.T])))
            self.seg_dur = self.beta[1:] - self.beta[:-1]
            self.T0 = self.beta[0]
            self.itvls = boundaries_to_intervals(self.beta)
            self.anno = mirevalflat2openseg(self.itvls, self.labels)
            self.levels = [self] # For consistency, a flat segmentation has one level
            self.d = 1

        # Common attributes
        self._Bhat = None
        self.Bhat_bw = None
        self.sr = None
        self.ticks = None
        if sr:
            self.update_sr(sr)
        if Bhat_bw is not None:
            self.update_bw(Bhat_bw)

    def update_bw(self, bw):
        if self.Bhat_bw == bw:
            return
        self.Bhat_bw = bw
        if self.is_hierarchical:
            for level in self.levels:
                level.update_bw(bw)
        else:
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
        if self.sr == sr:
            return
        self.sr = float(sr)
        if self.is_hierarchical:
            for level in self.levels:
                level.update_sr(sr)
            self.ticks = self.levels[0].ticks
        else:
            from mir_eval.hierarchy import _round
            frame_size = 1.0 / self.sr
            n_frames = int(
                (_round(self.T, frame_size) - _round(self.T0, frame_size)) / frame_size
            )
            self.ticks = np.arange(n_frames + 1) * frame_size + self.T0

    def B(self, x):
        if self.is_hierarchical:
            # For hierarchical, return coarsest level where x is a boundary
            for k in range(self.d):
                if self.levels[k].B(x): # Call B on the S-like level
                    return self.d - k
            return 0
        else: # Flat
            # TODO: Consider renaming `B` to something more descriptive like `has_boundary_at(x)`
            #       or `get_boundary_indicator(x)`.
            return int(x in self.beta or (x == self.T0) or (x == self.T))

    def Bhat(self, ts=None, weights=None):
        # TODO: Consider renaming `Bhat` to `boundary_salience_curve` or `smoothed_boundary_strength`.
        if ts is None:
            if self.ticks is None and self.sr is not None: # Ensure ticks are computed if sr is available
                 self.update_sr(self.sr)
            ts = self.ticks
        ts = np.array(ts)

        if self.Bhat_bw is None:
            warnings.warn("Bhat_bw is not set. setting it to 1.")
            self.update_bw(1)

        if self.is_hierarchical:
            if weights is None:
                weights = np.ones(self.d)
            weights /= np.sum(weights)
            # Bhats needs to be called on self to get list of Bhats from levels
            weighted = np.array(weights).reshape(-1, 1) * self.Bhats(ts)
            return np.sum(weighted, axis=0)
        else: # Flat
            if self._Bhat is None: # Should have been initialized by update_bw
                return np.zeros_like(ts, dtype=float)
            return self._Bhat(ts)

    def A(self, bs=None, compare_fn=np.equal):
        if bs is None:
            bs = self.beta # self.beta is union of all betas for hierarchical
        bs = np.array(sorted(set(bs)))

        if self.is_hierarchical:
            # Sum of A matrices from all levels
            # TODO: Consider renaming `A` to `label_agreement_indicator` or similar.
            return sum(level.A(bs=bs, compare_fn=compare_fn) for level in self.levels)
        else: # Flat
            ts = (bs[1:] + bs[:-1]) / 2
            labels_arr = np.array([self(t) for t in ts])
            return compare_fn.outer(labels_arr, labels_arr).astype(int)

    def Ahat(self, bs=None, compare_fn=np.equal, weights=None):
        # TODO: Consider renaming `Ahat` to `normalized_label_agreement_matrix` or similar.
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs)))

        if self.is_hierarchical:
            if weights is None:
                weights = np.ones(self.d)
            weights /= np.sum(weights)
            # Ahats needs to be called on self to get list of Ahats from levels
            weighted = np.array(weights).reshape(-1, 1, 1) * self.Ahats(bs=bs, compare_fn=compare_fn)
            return np.sum(weighted, axis=0)
        else: # Flat
            lai = self.A(bs=bs, compare_fn=compare_fn) # Call S-like A
            seg_dur = bs[1:] - bs[:-1]
            seg_dur_area_mat = np.outer(seg_dur, seg_dur)
            total_area = np.sum(seg_dur_area_mat * lai)
            return lai / total_area if total_area > 0 else lai

    def plot(self, axs=None, text=True, legend=False, legend_offset=0.2, time_ticks=True, **kwargs):
        if self.is_hierarchical:
            fig_kw = dict(
                figsize=(5, 0.4 * self.d + 0.5),
                h_ratios=[1] * self.d,
                w_ratios=[1],
            )
            fig_kw.update(kwargs)

            if axs is None:
                _, axs = viz.create_fig(**fig_kw)
                axs = np.array(axs).flatten()

            if len(axs) < self.d:
                raise ValueError(f"Number of axes ({len(axs)}) is smaller than number of levels ({self.d}).")

            style_map = viz.label_style_dict(labels=self.labels) # self.labels is list of lists for H

            if self.d > 1:
                for i in range(self.d - 1):
                    self.levels[i].plot( # Call plot on the S-like level
                        ax=axs[i],
                        ytick=i + 1,
                        time_ticks=False,
                        text=text,
                        style_map=style_map,
                        #figsize=None # Avoid passing figsize to S-level plot
                    )
            self.levels[-1].plot( # Call plot on the S-like level
                ax=axs[-1],
                ytick=self.d,
                time_ticks=time_ticks,
                text=text,
                style_map=style_map,
                #figsize=None # Avoid passing figsize to S-level plot
            )
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
        else: # Flat
            average_seg_dur = np.mean(self.seg_dur) if len(self.seg_dur) > 0 else 1.0
            atom_size = (average_seg_dur + (np.min(self.seg_dur) if len(self.seg_dur) > 0 else 1.0) ) / 2.0
            num_atoms = (np.sum(self.seg_dur) / atom_size) if atom_size > 0 else 1.0
            default_width = min(max(3.5, num_atoms * 0.35), 12)

            new_kwargs = dict(text=True, ytick="", time_ticks=True, figsize=(default_width, 0.5))
            # kwargs might include 'ax' from a hierarchical call, which should be prioritized or handled.
            # viz.segment takes 'ax' as a direct keyword argument, not through **new_kwargs if already explicit.

            explicit_ax_from_kwargs = kwargs.pop('ax', None) # Remove ax from kwargs if it was passed down
            new_kwargs.update(kwargs) # Update with remaining kwargs

            current_ax = axs[0] if isinstance(axs, (list, np.ndarray)) else axs
            current_ax = explicit_ax_from_kwargs if explicit_ax_from_kwargs is not None else current_ax

            if current_ax is None:
                # Only pop figsize if we are creating the figure here
                fig = plt.figure(figsize=new_kwargs.pop("figsize"))
                current_ax = fig.add_subplot(111)
            else:
                # If ax is provided, don't use figsize from new_kwargs
                new_kwargs.pop("figsize", None)

            return viz.segment(self.itvls, self.labels, ax=current_ax, **new_kwargs)

    def expand(self, format="slm", always_include=False):
        if self.is_hierarchical:
            expanded_levels_S_instances = []
            for level_S_instance in self.levels: # Iterate over S instances in H
                # Expand each S instance (which returns an H-like Segmentation)
                expanded_H_segmentation = level_S_instance.expand(format=format, always_include=always_include)
                expanded_levels_S_instances.extend(expanded_H_segmentation.levels) # Collect S instances
            return levels_to_segmentation(expanded_levels_S_instances, sr=self.sr, Bhat_bw=self.Bhat_bw)
        else: # Flat
            # This part is similar to S.expand, but returns a new Segmentation instance
            from .hierarchy import expand_hierarchy # Import here to break circular dependency
            expanded_jams_annos = expand_hierarchy(
                self.anno, dataset=format, always_include=always_include
            )
            # Convert each JAMS annotation back to a flat Segmentation (S-like)
            expanded_S_levels = [flat_to_segmentation(l, sr=self.sr, Bhat_bw=self.Bhat_bw) for l in expanded_jams_annos]
            # An expanded flat segmentation should always be returned as hierarchical
            return levels_to_segmentation(expanded_S_levels, sr=self.sr, Bhat_bw=self.Bhat_bw)


    def __call__(self, x):
        if self.is_hierarchical:
            return [level(x) for level in self.levels]
        else: # Flat
            if not (self.T0 <= x <= self.T):
                raise IndexError(f"RANGE: {x} outside the range of this segmentation {self.T0, self.T}!")
            elif x == self.T:
                return self.Lstar[self.beta[-2]]
            idx = np.searchsorted(self.beta, x, side="right") - 1
            return self.Lstar[self.beta[idx]]

    # --- Methods specific to Hierarchical Segmentation ---
    def _check_hierarchical(self, method_name):
        if not self.is_hierarchical:
            raise TypeError(f"Method '{method_name}' can only be called on a hierarchical segmentation.")

    def Ahats(self, bs=None, compare_fn=np.equal):
        self._check_hierarchical("Ahats")
        return np.asarray([lvl.Ahat(bs=bs, compare_fn=compare_fn) for lvl in self.levels]) # Call Ahat on S-like levels

    def Bhats(self, ts=None):
        self._check_hierarchical("Bhats")
        if self.Bhat_bw is None: # Ensure Bhat_bw is set for all levels
            warnings.warn("Bhat_bw is not set. setting it to 1.")
            self.update_bw(1)
        return np.asarray([lvl.Bhat(ts=ts) for lvl in self.levels]) # Call Bhat on S-like levels

    def Astar(self, bs=None):
        self._check_hierarchical("Astar")
        if bs is None: bs = self.beta
        bs = np.array(sorted(set(bs)))
        # Ahats call here will correctly get Ahat from each S-like level
        Ahats_matrices = self.Ahats(bs=bs)
        indexed_Ahats = np.array(
            [(level_idx + 1) * (Ahats_matrices[level_idx] > 0).astype(int) for level_idx in range(self.d)]
        )
            # TODO: Consider renaming `Astar` to `deepest_consistent_label_level` or `annotation_meet_matrix`.
        return np.max(indexed_Ahats, axis=0)

    def M(self, bs=None, level_weights=None):
        # TODO: Consider renaming `M` to `resampled_agreement_area_matrix` or similar.
        self._check_hierarchical("M")
        if bs is None: bs = self.beta
        bs = np.array(sorted(set(bs)))
        all_bs = np.array(sorted(set(self.beta).union(bs)))
        seg_dur = all_bs[1:] - all_bs[:-1]
        seg_agreement_area = np.outer(seg_dur, seg_dur)
        # Ahat call on self (H-like) will use the weighted sum of Ahats from S-like levels
        return utils.resample_matrix(
            seg_agreement_area * self.Ahat(bs=all_bs, weights=level_weights), all_bs, bs
        )

    def Mhat(self, bs=None, level_weights=None):
        self._check_hierarchical("Mhat")
        if bs is None: bs = self.beta
        bs = np.array(sorted(set(bs)))
        seg_dur = bs[1:] - bs[:-1]
        # M call on self (H-like)
        # TODO: Consider renaming `Mhat` to `normalized_resampled_agreement_area_matrix`.
        return self.M(bs=bs, level_weights=level_weights) / np.outer(seg_dur, seg_dur)

    def decode_B(self, depth=None, pre_max=0.8, post_max=0.8, pre_avg=0.3, post_avg=0.3, delta=1e-3, wait=1, level_weights=None, sr=None, bw=None):
        # TODO: Consider renaming `decode_B` to `decode_boundaries_from_salience` or similar.
        self._check_hierarchical("decode_B")
        if sr is not None: self.update_sr(sr)
        if bw is not None: self.update_bw(bw)
        if self.sr is None or self.Bhat_bw is None:
            raise ValueError("sr and bw must be set for decode_B")

        # Bhat call on self (H-like) will use weighted Bhats from S-like levels
        novelty = self.Bhat(ts=self.ticks, weights=level_weights)
        novelty /= novelty.max() + 1e-10
        novelty[0], novelty[-1] = novelty.max(), novelty.max()

        boundaries = librosa.util.peak_pick(novelty, pre_max=int(pre_max * self.sr), post_max=int(post_max * self.sr), pre_avg=int(pre_avg * self.sr), post_avg=int(post_avg * self.sr), delta=delta, wait=int(wait * self.sr))
        boundaries = np.unique(np.concatenate(([0, len(novelty) - 1], boundaries)))

        if depth is None: depth = self.d
        intervals = utils.cluster_boundaries(boundaries, novelty, self.ticks, depth)
        return intervals

    def decode_L(self, itvls, min_k=2):
        self._check_hierarchical("decode_L")
        current_k = min_k
        labs = []
        for lvl_itvls in itvls:
            bs = intervals_to_boundaries(lvl_itvls)
            # Mhat call on self (H-like)
            M_matrix = self.Mhat(bs=bs)
            lab, current_k = utils.eigen_gap_scluster(M_matrix, min_k=current_k)
            labs.append(lab)
        # Return a new H-like Segmentation
        # TODO: Consider renaming `decode_L` to `decode_labels_for_intervals` or similar.
        return Segmentation(itvls, labs, sr=self.sr, Bhat_bw=self.Bhat_bw, is_hierarchical=True)

    def decode(self, depth=4, min_k=2, **kwargs):
        # TODO: Consider renaming `decode` to `decode_full_hierarchy` or similar.
        self._check_hierarchical("decode")
        new_itvls = self.decode_B(depth=depth, **kwargs)
        return self.decode_L(new_itvls, min_k=min_k)


def levels_to_segmentation(levels_data, sr=None, Bhat_bw=None):
    """Converts a list of level data (S-like Segmentations or raw itvls/labels) to a new Segmentation instance."""
    if not levels_data: # Handle empty list
        return Segmentation([], [], sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)

    # If levels_data are already Segmentation instances (S-like)
    if isinstance(levels_data[0], Segmentation) and not levels_data[0].is_hierarchical:
        itvls = [l.itvls for l in levels_data]
        lbls = [l.labels for l in levels_data]
        return Segmentation(itvls, lbls, sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)
    # If levels_data is a list of (itvls, labels) tuples or similar raw data for S
    elif isinstance(levels_data[0], (tuple, list)) and len(levels_data[0]) == 2:
         #This case might need more robust checking depending on expected raw data format
        itvls = [l[0] for l in levels_data]
        lbls = [l[1] for l in levels_data]
        return Segmentation(itvls, lbls, sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)
    else: # Assuming it's already in the format [itvls_list, labels_list] or just itvls_list for H
        # This path might need refinement based on how raw H data is passed
        if len(levels_data) == 2 and isinstance(levels_data[0], list) and isinstance(levels_data[1], list): # itvls, labels
             return Segmentation(levels_data[0], levels_data[1], sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)
        elif isinstance(levels_data, list) and all(isinstance(lvl, (list, np.ndarray)) for lvl in levels_data): # only itvls
             return Segmentation(levels_data, None, sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)
    raise ValueError("Invalid format for levels_data in levels_to_segmentation")


def multi_to_segmentation(anno, sr=None, Bhat_bw=None):
    """Convert multiple segments (from JAMS annotation) to a hierarchical Segmentation."""
    itvls, labels = multi2mireval(anno) # This returns itvls_list, labels_list
    return Segmentation(itvls, labels, sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)


def flat_to_segmentation(anno, sr=None, Bhat_bw=None):
    """Convert a flat JAMS annotation to a flat Segmentation."""
    itvls, labels = openseg2mirevalflat(anno) # This returns itvls_array, labels_array
    return Segmentation(itvls, labels, sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=False)


def sal_to_segmentation(bs_sal, sr=None, Bhat_bw=None):
    """Convert salience dictionary to a hierarchical Segmentation."""
    boundaries_by_level_itvls = []
    if not bs_sal: # Handle empty salience dictionary
        return Segmentation([], [], sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)

    total_levels = max(bs_sal.values()) if bs_sal else 0
    for level_salience_threshold in range(total_levels, 0, -1):
        level_boundaries_times = [b_time for b_time, sal in bs_sal.items() if sal >= level_salience_threshold]
        if not level_boundaries_times: # Ensure there's at least start and end if no boundaries meet threshold
            # This case needs careful handling: what should be the itvls if no boundaries qualify?
            # For now, let's assume if bs_sal is not empty, it implies a duration.
            # A robust solution might require T_end to be passed or inferred.
            # If T_end is unknown, an empty interval list might be an option, or a single [0, T_end] if T_end is known.
            # For simplicity, let's assume this implies an empty level if no boundaries are found.
             pass # Or append an empty list/ appropriately handled interval
        else:
            boundaries_by_level_itvls.append(boundaries_to_intervals(sorted(level_boundaries_times)))

    # Filter out empty levels before creating Segmentation to avoid issues with S expecting non-empty itvls
    valid_levels_itvls = [itvls for itvls in boundaries_by_level_itvls if len(itvls) > 0]
    if not valid_levels_itvls and bs_sal: # If all levels ended up empty but bs_sal was not, create a single level with overall bounds
        all_bs = sorted(bs_sal.keys())
        if all_bs:
            valid_levels_itvls = [boundaries_to_intervals([all_bs[0], all_bs[-1]])] if len(all_bs) >=2 else [boundaries_to_intervals([0, all_bs[0]])] if len(all_bs)==1 else []


    # The H constructor in the original code had a prune_identical_levels() call.
    # This logic might need to be re-integrated here or as a method in Segmentation.
    # For now, creating the Segmentation directly:
    # We need labels for these itvls, default H behavior was [[str(s) for s in itvl[:,0]] for itvl in itvls]
    temp_seg = Segmentation(valid_levels_itvls, None, sr=sr, Bhat_bw=Bhat_bw, is_hierarchical=True)
    # Add prune_identical_levels equivalent if necessary, e.g. temp_seg.prune_identical_levels()
    # This method is not yet defined in Segmentation class.
    # For now, returning temp_seg. The prune logic should be added to Segmentation if needed.
    return temp_seg # .prune_identical_levels() - this method needs to be added to Segmentation
