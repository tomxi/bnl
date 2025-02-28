# I want to plot some meet mats for hier annotations returned by mir_eval
# I want to see where the hits and the misses are between two meet matrix.

import bnl, tests
import mir_eval, os, librosa
import matplotlib.pyplot as plt
import numpy as np


def make_hierarchies():
    hier1 = bnl.H([tests.ITVLS1, tests.ITVLS2], [tests.LABELS1, tests.LABELS2])
    hier2 = bnl.H(
        [tests.ITVLS3, tests.ITVLS4, tests.ITVLS5],
        [tests.LABELS3, tests.LABELS4, tests.LABELS5],
    )
    return dict(h1=hier1, h2=hier2)


def make_meet_mat(hier, frame_size=0.1, strict_mono=False):
    """Create meet matrices for a hierarchy with given frame size."""
    hier.update_sr(1 / frame_size)
    return mir_eval.hierarchy._meet(
        hier.itvls, hier.labels, frame_size, strict_mono=strict_mono
    )


def frame_gauc(meet_mat_ref, meet_mat_est):
    """
    Compute ranking recall and normalizer for each query position.

    Parameters:
    -----------
    meet_mat_ref : scipy.sparse matrix
        Reference meet matrix
    meet_mat_est : scipy.sparse matrix
        Estimated meet matrix

    Returns:
    --------
    q_ranking_recall : numpy.ndarray
        Ranking recall for each query position
    q_ranking_normalizer : numpy.ndarray
        Normalizer for each query position
    """
    q_ranking_recall = np.zeros(meet_mat_ref.shape[0])
    q_ranking_normalizer = np.zeros(meet_mat_ref.shape[0])

    for q in range(meet_mat_ref.shape[0]):
        # get the q'th row
        q_relevance_ref = np.delete(meet_mat_ref.getrow(q).toarray().ravel(), q)
        q_relevance_est = np.delete(meet_mat_est.getrow(q).toarray().ravel(), q)
        # count ranking violations
        inversions, normalizer = mir_eval.hierarchy._compare_frame_rankings(
            q_relevance_ref, q_relevance_est, transitive=True
        )
        q_ranking_recall[q] = (1.0 - inversions / normalizer) if normalizer else 0
        q_ranking_normalizer[q] = normalizer

    agg_recall = np.mean(q_ranking_recall[np.where(q_ranking_normalizer != 0)])

    return agg_recall, q_ranking_recall, q_ranking_normalizer


def lmeasure_comparison(ref, est, frame_size=0.1):
    """Compare per-frame lmeasure of two hierarchies by computing ranking recall and precision
    for both non-monotonic and monotonic meet matrices.
    """
    results = {}
    for mode in ("orig", "mono"):
        strict_mono = True if mode == "mono" else False
        mat_ref = make_meet_mat(ref, frame_size=frame_size, strict_mono=strict_mono)
        mat_est = make_meet_mat(est, frame_size=frame_size, strict_mono=strict_mono)
        recall, rank_recall, norm_recall = frame_gauc(mat_ref, mat_est)
        precision, rank_precision, norm_precision = frame_gauc(mat_est, mat_ref)
        results[mode] = {
            "ref_meet": mat_ref,
            "est_meet": mat_est,
            "l": (precision, recall),
            "q": (rank_precision, rank_recall),
            "norm": (norm_precision, norm_recall),
        }
    return results


def plot_comparison(ref, est, frame_size=0.5):
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


if __name__ == "__main__":
    hier_ref, hier_est = list(make_hierarchies().values())
    fig, axs = plot_comparison(hier_ref, hier_est, frame_size=0.1)
    fig.savefig("scripts/figs/meet_mats_compare_both.pdf")
