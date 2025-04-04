import bnl
from bnl import fio, viz, H, S
from mir_eval import hierarchy
import numpy as np


def make_hierarchies():
    ITVLS1 = np.array([[0, 2.5], [2.5, 6.01]])
    LABELS1 = ["A", "B"]

    ITVLS2 = np.array([[0, 1.2], [1.2, 2.5], [2.5, 3.5], [3.5, 6.01]])
    LABELS2 = ["a", "b", "c", "b"]

    ITVLS3 = np.array([[0, 1.2], [1.2, 4], [4, 6.01]])
    LABELS3 = ["Mi", "Re", "Do"]

    ITVLS4 = np.array([[0, 1.2], [1.2, 3], [3, 4], [4, 6.01]])
    LABELS4 = ["T", "PD", "D", "T"]

    ITVLS5 = np.array(
        [[0, 1.2], [1.2, 2], [2, 3], [3, 4], [4, 4.7], [4.7, 5.3], [5.3, 6.01]]
    )
    LABELS5 = ["I", "IV", "ii", "V", "I", "IV", "I"]

    hier1 = H([ITVLS1, ITVLS2], [LABELS1, LABELS2])
    hier2 = H([ITVLS3, ITVLS4, ITVLS5], [LABELS3, LABELS4, LABELS5])
    hier3 = H(
        [ITVLS1, ITVLS2, ITVLS3, ITVLS4, ITVLS5],
        [LABELS1, LABELS2, LABELS3, LABELS4, LABELS5],
    )
    hier1.update_sr(25)
    hier2.update_sr(25)
    hier3.update_sr(25)
    return dict(h1=hier1, h2=hier2, h3=hier3)


def plot_column(hier, q_fraq=0.3, axs=None):
    ts = (hier.ticks[:-1] + hier.ticks[1:]) / 2
    q = int(q_fraq * (len(ts) - 1))
    q2 = int((1 - q_fraq) * (len(ts) - 1))

    axs = np.asarray(axs).flatten()

    # Let's figure out if we need to turn off some axis.
    # This is going to leave the first rows empty for short hierarchies.
    num_rows = hier.d + 3
    empty_rows = len(axs) - num_rows
    # turn off empty row axis and adjust y for title
    y = 1.0
    for i in range(empty_rows):
        axs[i].axis("off")
        y -= 1
    # Let's plot the hierarchy
    hier.plot(axs=axs[empty_rows : empty_rows + hier.d], text=False, time_ticks=False)
    axs[0].set_title("Hierarchy", y=y)

    next_row = empty_rows + hier.d
    # Now plot the meet matrix on axs[hier.d+1]
    meet = hierarchy._meet(hier.itvls, hier.labels, frame_size=1.0 / hier.sr).toarray()
    viz.sq(meet, hier.ticks, ax=axs[next_row], cmap="gray_r")
    axs[next_row].set_title("Label Agreement Map $M$")
    # add crosshair at q
    axs[next_row].vlines(
        ts[q], ts[0], ts[-1], colors="r", linestyles="dashed", label="Query time"
    )
    axs[next_row].hlines(ts[q], ts[0], ts[-1], colors="r", linestyles="dashed")
    axs[next_row].legend()

    # Plot relevance at q on axs[-2]
    axs[next_row + 1].sharex(axs[next_row])
    q_rel = meet[q]
    axs[next_row + 1].plot(ts, q_rel, color="k")
    axs[next_row + 1].fill_between(ts, q_rel, color="pink", alpha=0.8)
    axs[next_row + 1].vlines(
        ts[q], 0, max(q_rel), colors="r", linestyles="dashed", label="Query time"
    )
    axs[next_row + 1].set(
        title="Relevance to t: $M(t,\cdot)$", xlabel="Time (s)", ylabel="Meet Depth"
    )
    axs[next_row + 1].legend()

    # Relevant pairs u, v for frame q on axs[next_row + 3]
    u_more_relevant_mat = np.less.outer(q_rel, q_rel)
    viz.sq(u_more_relevant_mat, hier.ticks, ax=axs[next_row + 2], cmap="Reds")
    axs[next_row + 2].set(
        title="Relevant Triplets $\mathcal{A}(H;t)$", xlabel="u", ylabel="v"
    )
    fig = axs[0].get_figure()
    return fig, axs, u_more_relevant_mat


if __name__ == "__main__":
    hiers = make_hierarchies()
    h1 = hiers["h1"]
    h2 = hiers["h2"]
    h3 = hiers["h3"]
    max_d = max(h1.d, h2.d)
    fig, axs = viz.create_fig(
        w_ratios=[1, 1],
        h_ratios=[0.5] * max_d + [10, 2.5, 10, 10],
        figsize=(5, 9),
        h_gaps=[0.001] * (max_d - 1) + [0.001] * 4,
    )

    fig, _, r_sig_pairs = plot_column(h1, axs=axs[:-1, 0])
    fig, _, e_sig_pairs = plot_column(h2, axs=axs[:-1, 1])

    intersect_pairs = (r_sig_pairs * e_sig_pairs).astype(int)
    false_negative_pairs = -intersect_pairs + r_sig_pairs.astype(int)
    false_positive_pairs = -intersect_pairs + e_sig_pairs.astype(int)

    axs[0, 0].set_title("Reference Hierarchy $H$")
    axs[0, 1].set_title("Estimated Hierarchy $\hat{H}$")
    axs[-1, 0].set_title("Intersection and Difference")
    filtered_ref = intersect_pairs - false_negative_pairs
    viz.sq(filtered_ref, h1.ticks, ax=axs[-1, 0])
    axs[-1, 0].set(xlabel="u", ylabel="v")
    axs[-1, 1].set_title("Intersection and Difference")
    filtered_est = intersect_pairs - false_positive_pairs
    viz.sq(filtered_est, h2.ticks, ax=axs[-1, 1])
    axs[-1, 1].set(xlabel="u", ylabel="v")
    axs[-2, -1].set_title("Relevant Triplets $\mathcal{A}(\hat{H};t)$")

    # Make the bottom three rows' axis box be in red dashed lines
    for ax in axs[-3:, :].flatten():
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linestyle("dashed")
            spine.set_linewidth(1.5)

    for ax in axs.flat[6:]:
        ax.label_outer()

    # now compute the intersection of the two hierarchies relevant pair
    fig.savefig("scripts/explain_triplet.pdf", transparent=True)
