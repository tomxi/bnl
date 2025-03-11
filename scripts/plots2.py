import bnl, tests
from bnl import fio, viz
from mir_eval import hierarchy
import numpy as np


def make_fig(hier, q_fraq=0.3):
    ts = (hier.ticks[:-1] + hier.ticks[1:]) / 2
    q = int(q_fraq * (len(ts) - 1))

    fig, axs = viz.create_fig(
        w_ratios=[1],
        h_ratios=[0.8, 0.8, 0.8, 12, 3, 12],
        figsize=(5, 15.5),
        h_gaps=[0.001] * 3 + [0.5] * 2,
    )

    axs = np.array(axs).flatten()
    # for ax in axs[1:]:
    #     ax.sharex(axs[0])
    axs[4].sharex(axs[3])

    # Let's plot the 3 level hierarchy
    hier.plot(axs=axs[:3])
    axs[0].set_title("Hierarchical Segmentation")

    # Now plot the meet matrix on axs[3]
    meet = hierarchy._meet(hier.itvls, hier.labels, frame_size=1.0 / hier.sr).toarray()
    viz.sq(meet, hier.ticks, ax=axs[3], cmap="gray_r")
    axs[3].set_title("Meet matrix")
    # add crosshair at q
    axs[3].vlines(
        ts[q], ts[0], ts[-1], colors="r", linestyles="dashed", label="Query Frame"
    )
    axs[3].hlines(ts[q], ts[0], ts[-1], colors="r", linestyles="dashed")
    axs[3].legend()

    # Plot relevance at q on axs[4]
    q_rel = meet[q]
    axs[4].plot(ts, q_rel, color="k")
    axs[4].vlines(
        ts[q], 0, max(q_rel), colors="r", linestyles="dashed", label="Query Frame"
    )
    axs[4].legend()
    axs[4].set(title="Relevance to q", xlabel="Time (s)", ylabel="Relevance")

    # Relevant pairs u, v for frame q on axs[5]
    u_more_relevant_mat = np.greater.outer(q_rel, q_rel)
    v_more_relevant_mat = np.less.outer(q_rel, q_rel)
    combined_mat = u_more_relevant_mat.astype(int) - v_more_relevant_mat.astype(int)
    viz.sq(combined_mat, hier.ticks, ax=axs[5])
    axs[5].set(title="Relevant Pairs (u, v)", xlabel="Frame u", ylabel="Frame v")

    return fig, axs


def plot_column(hier, q_fraq=0.3, axs=None):
    ts = (hier.ticks[:-1] + hier.ticks[1:]) / 2
    q = int(q_fraq * (len(ts) - 1))

    if axs is None:
        _, axs = viz.create_fig(
            w_ratios=[1],
            h_ratios=[0.8] * hier.d + [12, 3, 12],
            figsize=(5, 15.5),
            h_gaps=[0.001] * (hier.d - 1) + [0.5] * 3,
        )
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
    hier.plot(axs=axs[empty_rows : empty_rows + hier.d])
    axs[0].set_title("Hierarchical Segmentation", y=y)

    # Now plot the meet matrix on axs[hier.d+1]
    meet = hierarchy._meet(hier.itvls, hier.labels, frame_size=1.0 / hier.sr).toarray()
    viz.sq(meet, hier.ticks, ax=axs[-3], cmap="gray_r")
    axs[-3].set_title("Meet matrix")
    # add crosshair at q
    axs[-3].vlines(
        ts[q], ts[0], ts[-1], colors="r", linestyles="dashed", label="Query Frame"
    )
    axs[-3].hlines(ts[q], ts[0], ts[-1], colors="r", linestyles="dashed")
    axs[-3].legend()

    # Plot relevance at q on axs[-2]
    axs[-2].sharex(axs[hier.d])
    q_rel = meet[q]
    axs[-2].plot(ts, q_rel, color="k")
    axs[-2].vlines(
        ts[q], 0, max(q_rel), colors="r", linestyles="dashed", label="Query Frame"
    )
    axs[-2].set(title="Relevance to q", xlabel="Time (s)", ylabel="Relevance")
    axs[-2].legend()

    # Relevant pairs u, v for frame q on axs[hier.d + 3]
    u_more_relevant_mat = np.greater.outer(q_rel, q_rel)
    v_more_relevant_mat = np.less.outer(q_rel, q_rel)
    combined_mat = u_more_relevant_mat.astype(int) - v_more_relevant_mat.astype(int)
    viz.sq(combined_mat, hier.ticks, ax=axs[-1])
    axs[-1].set(title="Relevant Pairs (u, v)", xlabel="Frame u", ylabel="Frame v")
    fig = axs[0].get_figure()
    return fig, axs


if __name__ == "__main__":
    hiers = tests.make_hierarchies()
    h1 = hiers["h1"]
    h2 = hiers["h2"]
    h3 = hiers["h3"]
    max_d = max(h1.d, h2.d)
    fig, axs = viz.create_fig(
        w_ratios=[1, 1],
        h_ratios=[0.8] * max_d + [10, 2, 10],
        figsize=(8, 12),
        h_gaps=[0.001] * (max_d - 1) + [0.001] * 3,
    )

    fig, _ = plot_column(h1, axs=axs[:, 0])
    fig, _ = plot_column(h2, axs=axs[:, 1])
    fig.savefig("scripts/figs/fig2.pdf", transparent=True)
