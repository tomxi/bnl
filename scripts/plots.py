# I want to plot some meet mats for hier annotations returned by mir_eval
# I want to see where the hits and the misses are between two meet matrix.


import bnl, tests
from bnl.utils import gauc
from bnl import fio, viz
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


def lmeasure_comparison(ref: bnl.H, est: bnl.H, frame_size=0.1):
    """Compare per-frame lmeasure of two hierarchies by computing ranking recall and precision
    for both non-monotonic and monotonic meet matrices.
    """
    results = dict(orig={}, mono={})
    ## Make sure ref and est have the same sr: 1 / frame_size
    ref.update_sr(1 / frame_size)
    est.update_sr(1 / frame_size)

    for mode in results.keys():
        strict_mono = True if mode == "mono" else False
        mat_ref = make_meet_mat(ref, strict_mono=strict_mono)
        mat_est = make_meet_mat(est, strict_mono=strict_mono)
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


# def fig_hiers(hiers):
#     fig, axs = plt.subplots(1, len(hiers), figsize=(12, 4))
#     for i, h in enumerate(hiers):
#         h.plot(ax=axs[i], text=True, relabel=False, legend_ncol=None)
#     return fig, axs


def fig_a_meet_mats(h, figsize=(8, 3)):
    # I want to plot annotation and meet mats for flat, hierarchical, and boundary
    # make a 1 row 3 column grid, with the first row for the annotation, the second row for the meet mat
    fig, axs = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex="all",
        sharey="row",
    )

    hiers = [bnl.levels2H([h.levels[1]]), h.unique_labeling(), h]

    titles = ["Flat: Level 2", "Hierarchy: Boundary", "Hierarchy: Label"]
    for i, title in enumerate(titles):
        axs[i].set_title(title)
        h = hiers[i]
        meet_mat = mir_eval.hierarchy._meet(
            h.itvls, h.labels, frame_size=1.0 / h.sr
        ).toarray()
        # Plot Meet mat on axs[i]
        quadmesh = viz.square(meet_mat, h.ticks, axs[i], cmap="gray_r")
        if i != 0:
            axs[i].set_ylabel("")
    fig.tight_layout()
    return fig, axs


def fig_b_frame_q_recall(h, q_frac=0.3, figsize=(8, 3)):
    """I want to plot a meet mat, overlay with hline and vline at query frame q.
    Then look at the row at q, and plot the row as a curve, with q marked as a vline
    Finally, find all significant pairs u,v where the relevence (q, u) and (q, v) are different.

    q_frac is [0, 1], indicate where the query frame q is along the whole duration of the hierarchy.
    """
    # first make a 3 row subplot with 4, 1, 4 row height
    fig, axs = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex="all",
        sharey="row",
        gridspec_kw={"height_ratios": [4, 1, 4]},
    )
    ts = (h.ticks[:-1] + h.ticks[1:]) / 2
    q = int(q_frac * len(ts))

    # Plot the meet mat on the first row
    meet_mat = mir_eval.hierarchy._meet(
        h.itvls, h.labels, frame_size=1.0 / h.sr
    ).toarray()
    quadmesh = viz.square(meet_mat, h.ticks, axs[0], cmap="gray_r")
    axs[0].set_title("Meet Matrix")
    # Add vline and hline at query frame q

    axs[0].vlines(ts[q], ts[0], ts[-1], colors="r", linestyles="dashed")
    axs[0].hlines(ts[q], ts[0], ts[-1], colors="r", linestyles="dashed")

    # Plot the query frame q's relevance on the second row

    q_relevance = meet_mat[q, :]
    axs[1].set_title("Relevance to Query Frame")
    axs[1].plot(ts, q_relevance)
    axs[1].vlines(
        ts[q], 0, max(q_relevance), colors="r", linestyles="dashed", label="Query Frame"
    )
    axs[1].legend()

    # Plot the relevent pairs u, v according to the query frame q.
    # Find all significant pairs u, v where the relevance (q, u) and (q, v) are different
    rel_pair_mat = np.greater.outer(q_relevance, q_relevance)
    rel_pair_mat_neg = np.less.outer(q_relevance, q_relevance)

    librosa.display.specshow(
        rel_pair_mat.astype(float) - rel_pair_mat_neg.astype(float),
        ax=axs[2],
        x_coords=h.ticks,
        y_coords=h.ticks,
        x_axis="time",
        y_axis="time",
        cmap="coolwarm",
    )
    axs[2].set_title("Relevant Pairs for Query Frame")
    axs[2].set_xlabel("Frame u")
    axs[2].set_ylabel("Frame v")

    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    h_dict = make_hierarchies()
    # h3 = h_dict["h3"]
    # h2 = h_dict["h2"]
    # h1 = h_dict["h1"]

    replot_part_1 = True

    # Plot the 3 level hierarchy that's h_dict["h2"]
    if not os.path.exists("scripts/figs/hier_fig.pdf") or replot_part_1:
        seg_fig, _ = h_dict["h2"].plot(
            text=True,
            relabel=False,
            figsize=(5, 2.2),
            legend_ncol=None,
        )
        seg_fig.suptitle("3 Level Hierarchy")
        seg_fig.tight_layout(pad=0.2, rect=[0, 0, 0.93, 1])
        seg_fig.savefig("scripts/figs/hier_fig.pdf")
        plt.close()

    #
    if not os.path.exists("scripts/figs/hier_meets.pdf") or replot_part_1:
        fig, axs = fig_a_meet_mats(h_dict["h2"], figsize=(8, 3))
        fig.savefig("scripts/figs/hier_meets.pdf")
        plt.close()
        # Plot meet matrices for the 3 hierarchies

    replot_part_2 = True
    if not os.path.exists("scripts/figs/q_relevant_pairs.pdf") or replot_part_2:
        h = h_dict["h2"]
        h.update_sr(50)
        fig, axs = fig_b_frame_q_recall(h, q_frac=0.3, figsize=(4, 10))
        fig.savefig("scripts/figs/q_relevant_pairs.pdf")
        plt.close()

        fig, axs = fig_b_frame_q_recall(h, q_frac=0.8, figsize=(4, 10))
        fig.savefig("scripts/figs/q_relevant_pairs2.pdf")
        plt.close()

    # Let's see these examples for t-measures. The relevant pairs are more intelligible with t-measures

    # let's get a est
    mosaic = "1;2;3;m;n"
    fig, axs = plt.subplot_mosaic(
        mosaic, sharex=True, height_ratios=[1, 1, 1, 4, 4], gridspec_kw={"hspace": 0.05}
    )
    # no ticks for the first 3 rows
    for ax_name in axs:
        if ax_name in ["1", "2", "3"]:
            ax = axs[ax_name]
            ax.set_xticks([])
            ax.set_yticks([])
        elif ax_name in ["m", "n"]:
            pass
    viz.identify_axes(axs, fontsize=12)

    fig.savefig("scripts/figs/ax_mosaic.pdf")
    plt.close()
