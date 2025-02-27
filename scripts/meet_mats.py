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


def make_meet_mats(hier, frame_size=0.1):
    # Call mir_eval.hierarchy._meet to get the meet matrices for a hierarchy
    # gets both monotonous and non-monotonic meet matrices. Original first, strict later.
    return [
        mir_eval.hierarchy._meet(hier.itvls, hier.labels, frame_size, strict_mono=flag)
        for flag in (False, True)
    ]


def plot_meet_mats(hier, frame_size=0.1):
    # make sure the hier has the same sr as the frame size for mir_eval _meet
    hier.update_sr(1 / frame_size)
    meet_mats = make_meet_mats(hier, frame_size=frame_size)
    # Plot the meet matrices in a row and share the color bar
    fig, axs = plt.subplots(1, len(meet_mats), figsize=(8, 3))
    for ax, mat in zip(axs, meet_mats):
        im = librosa.display.specshow(
            mat.toarray(),
            x_axis="time",
            y_axis="time",
            hop_length=1,
            sr=1 / frame_size,
            ax=ax,
        )
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axs)
    cbar.set_ticks(range(int(mat.max()) + 1))
    return fig, axs


def save_test_meet_mats(out_dir="scripts/figs", frame_size=0.1):
    os.makedirs(out_dir, exist_ok=True)
    hiers = make_hierarchies()
    for name in hiers:
        fig, _ = plot_meet_mats(hiers[name], frame_size=frame_size)
        fig.savefig(os.path.join(out_dir, f"meet_mats_{name}.pdf"))
        plt.close(fig)


def next_step(idx=0):
    ## Now I have the two meet mats from _meet, follow alone with the mir_eval and get to the count inversion stage.
    ## Let's see which pixels are hitting and which pixels are missing.
    hier1, hier2 = make_hierarchies().values()
    meet_mat_ref = make_meet_mats(hier1, frame_size=0.5)[idx]
    meet_mat_est = make_meet_mats(hier2, frame_size=0.5)[idx]

    # Now we have the meet matrices, for each query position q, we want to see violations in relevance score rankings
    q_ranking_recall = np.zeros(meet_mat_ref.shape[0])
    for q in range(meet_mat_ref.shape[0]):
        # get the q'th row
        q_relevance_ref = np.delete(meet_mat_ref.getrow(q).toarray().ravel(), q)
        q_relevance_est = np.delete(meet_mat_est.getrow(q).toarray().ravel(), q)
        # count ranking violations
        inversions, normalizer = mir_eval.hierarchy._compare_frame_rankings(
            q_relevance_ref, q_relevance_est, transitive=True
        )
        q_ranking_recall[q] = 1.0 - inversions / normalizer
        # print(f"query idx {q}: {inversions} {normalizer}")

    # print(q_relevance_ref, q_relevance_est)
    print(f"Recall scores: {q_ranking_recall}")
    plt.plot(q_ranking_recall)
    plt.show()

    # So I would like to see, when switching between their strictly monotonic or original setup, where the recall measures are changing the most.
    # Let's get a curve over q for the recall scores.


if __name__ == "__main__":
    # save_test_meet_mats(out_dir="scripts/figs", frame_size=0.5)
    next_step(0)
    next_step(1)
