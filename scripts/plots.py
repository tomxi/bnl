# I want to plot some meet mats for hier annotations returned by mir_eval
# I want to see where the hits and the misses are between two meet matrix.


import bnl, tests
from bnl.utils import gauc
from bnl import fio
import mir_eval, librosa, os, jams, json
import matplotlib.pyplot as plt
import numpy as np


def make_hierarchies():
    hier1 = bnl.H([tests.ITVLS1, tests.ITVLS2], [tests.LABELS1, tests.LABELS2])
    hier2 = bnl.H(
        [tests.ITVLS3, tests.ITVLS4, tests.ITVLS5],
        [tests.LABELS3, tests.LABELS4, tests.LABELS5],
    )
    hier3 = bnl.H(
        [tests.ITVLS1, tests.ITVLS2, tests.ITVLS3, tests.ITVLS4, tests.ITVLS5],
        [tests.LABELS1, tests.LABELS2, tests.LABELS3, tests.LABELS4, tests.LABELS5],
    )
    return dict(h1=hier1, h2=hier2, h3=hier3)


def make_meet_mat(hier, frame_size=0.1, strict_mono=False):
    """Create meet matrices for a hierarchy with given frame size."""
    hier.update_sr(1 / frame_size)
    return mir_eval.hierarchy._meet(
        hier.itvls, hier.labels, frame_size, strict_mono=strict_mono
    )


def lmeasure_comparison(ref: bnl.H, est: bnl.H, frame_size=0.1):
    """Compare per-frame lmeasure of two hierarchies by computing ranking recall and precision
    for both non-monotonic and monotonic meet matrices.
    """
    results = dict(orig={}, mono={})
    for mode in results.keys():
        strict_mono = True if mode == "mono" else False
        mat_ref = make_meet_mat(ref, frame_size=frame_size, strict_mono=strict_mono)
        mat_est = make_meet_mat(est, frame_size=frame_size, strict_mono=strict_mono)
        recall, rank_recall, norm_recall = gauc(mat_ref, mat_est)
        precision, rank_precision, norm_precision = gauc(mat_est, mat_ref)
        results[mode] = {
            "ref_meet": mat_ref,
            "est_meet": mat_est,
            "l": (precision, recall),
            "q": (rank_precision, rank_recall),
            "norm": (norm_precision, norm_recall),
        }
    return results


def plot_comparison(ref: bnl.H, est: bnl.H, frame_size=0.5):
    """Compare per-frame lmeasure of two hierarchies (with or without strict monotonicity) and plot the results."""
    full_result = lmeasure_comparison(ref, est, frame_size=frame_size)
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(14, 6.5),
        sharex="all",
        sharey="row",
        gridspec_kw={"height_ratios": [4, 1, 1]},
    )

    # Iterate over both modes: original (non-strict) and strict monotonicity
    for strict_mono, offset in zip((False, True), (0, 2)):
        mode = "mono" if strict_mono else "orig"
        # Extract scores and compute time axis
        result = full_result[mode]
        lp, lr = result["l"]
        qp, qr = result["q"]
        norm_p, norm_r = result["norm"]
        ts = np.arange(len(qp)) * frame_size

        # Top row: Plot the reference and estimated meet matrices
        keys = ["ref_meet", "est_meet"]
        titles = [
            f"{mode} reference meet matrix",
            f"{mode} estimated meet matrix",
        ]
        for i, (key, title) in enumerate(zip(keys, titles)):
            ax = axs[0, offset + i]
            librosa.display.specshow(
                result[key].toarray(),
                ax=ax,
                x_axis="time",
                y_axis="time",
                hop_length=1,
                sr=1 / frame_size,
                cmap="gray_r",
            )
            ax.set_title(title)
            ax.set_xlabel("")
            # For non-mono, only clear the ylabel of the second plot
            # For mono, clear ylabels for both plots
            if offset or i:
                ax.set_ylabel("")

        # Second row: Plot norm curves
        axs[1, offset].plot(ts, norm_r)
        axs[1, offset].set_title("total significant pairs in reference")
        axs[1, offset + 1].plot(ts, norm_p)
        axs[1, offset + 1].set_title("total specificity in estimation")

        # Third row: Plot per-frame scores and add horizontal lines
        axs[2, offset].plot(ts, qr)
        axs[2, offset].set_title("per frame recall")
        axs[2, offset].set_xlabel("Time")
        axs[2, offset + 1].plot(ts, qp)
        axs[2, offset + 1].set_title("per frame precision")
        axs[2, offset + 1].set_xlabel("Time")

        axs[2, offset].hlines(
            lr,
            ts[0],
            ts[-1],
            linestyles="dashed",
            color="r",
            label=f"{lr:.2f}",
        )
        axs[2, offset].legend()
        axs[2, offset + 1].hlines(
            lp,
            ts[0],
            ts[-1],
            linestyles="dashed",
            color="r",
            label=f"{lp:.2f}",
        )
        axs[2, offset + 1].legend()

    fig.tight_layout()
    return fig, axs


def fig_hiers(hiers):
    fig, axs = plt.subplots(1, len(hiers), figsize=(12, 4))
    for i, h in enumerate(hiers):
        h.plot(ax=axs[i], text=True, relabel=False, legend_ncol=None)
    return fig, axs


def fig_meet_mats(h):
    # I want to plot annotation and meet mats for flat, hierarchical, and boundary
    # make a 1 row 3 column grid, with the first row for the annotation, the second row for the meet mat
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(8, 3),
        sharex="all",
        sharey="row",
    )

    hiers = [bnl.levels2H([h.levels[1]]), h.unique_labeling(), h]

    titles = ["Level 2 Flat", "Boundary Hierarchy", "Hierarchy"]
    for i, title in enumerate(titles):
        axs[i].set_title(title)
        h = hiers[i]
        meet_mat = make_meet_mat(h, strict_mono=False).toarray()
        # ticks = (h.ticks[:-1] + h.ticks[1:]) / 2.0
        # Plot Meet mat on axs[i]
        librosa.display.specshow(
            meet_mat,
            ax=axs[i],
            x_coords=h.ticks,
            y_coords=h.ticks,
            x_axis="time",
            y_axis="time",
            cmap="gray_r",
        )
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    h_dict = make_hierarchies()
    h3 = h_dict["h3"]
    h2 = h_dict["h2"]
    h1 = h_dict["h1"]

    recompute = True

    if not os.path.exists("scripts/figs/hier_fig.pdf") or recompute:
        seg_fig, _ = h2.plot(
            text=True,
            relabel=False,
            figsize=(5.5, 2.5),
            legend_ncol=None,
        )
        seg_fig.suptitle("3 Level Hierarchy")
        seg_fig.tight_layout(pad=0.2, rect=[0, 0, 0.93, 1])
        seg_fig.savefig("scripts/figs/hier_fig.pdf")
        plt.close()

    if not os.path.exists("scripts/figs/hier_meets.pdf") or True:
        fig, axs = fig_meet_mats(h2)
        fig.savefig("scripts/figs/hier_meets.pdf")
        plt.close()
        # Plot meet matrices for the 3 hierarchies
