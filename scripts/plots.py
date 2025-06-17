import bnl
from bnl import viz, H
from mir_eval import hierarchy
import mir_eval
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


def make_hierarchies():
    ITVLS1 = np.array([[0, 2.5], [2.5, 6.01]]) / 6.01 * 5.01
    LABELS1 = ["A", "B"]

    ITVLS2 = np.array([[0, 1.2], [1.2, 2.5], [2.5, 3.5], [3.5, 6.01]]) / 6.01 * 5.01
    LABELS2 = ["a", "b", "c", "b"]

    ITVLS3 = np.array([[0, 1.2], [1.2, 4], [4, 6.01]]) / 6.01 * 5.01
    LABELS3 = ["Mi", "Re", "Do"]

    ITVLS4 = np.array([[0, 1.2], [1.2, 3], [3, 4], [4, 6.01]]) / 6.01 * 5.01
    LABELS4 = ["T", "PD", "D", "T"]

    ITVLS5 = (
        np.array(
            [[0, 1.2], [1.2, 2], [2, 3], [3, 4], [4, 4.7], [4.7, 5.3], [5.3, 6.01]]
        )
        / 6.01
        * 5.01
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


def explain_triplet():
    """Generate the triplet explanation plot."""
    hiers = make_hierarchies()
    h1 = hiers["h1"]
    h2 = hiers["h2"]
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
    axs[-1, 0].set_title("Intersection")
    filtered_ref = intersect_pairs - false_negative_pairs
    viz.sq(filtered_ref, h1.ticks, ax=axs[-1, 0])
    axs[-1, 0].set(xlabel="u", ylabel="v")
    axs[-1, 1].set_title("Intersection")
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


def explain_pfc():
    """Generate the pairwise false positive/negative explanation plot."""
    import matplotlib.pyplot as plt

    hiers = list(bnl.fio.salami_ref_hiers(464).values())

    a = hiers[0].levels[0]
    b = hiers[1].levels[0]

    ## Create a plot of 3 rows and 2 columns
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(6, 7),
        gridspec_kw={"height_ratios": [0.8, 5, 5]},
        sharey="row",
        sharex="col",
    )

    a.plot(ax=axes[0, 0], text=False)
    b.plot(ax=axes[0, 1], text=False)
    bnl.viz.sq(a.A(), a.beta, ax=axes[1, 0], cmap="gray_r")
    bnl.viz.sq(b.A(), b.beta, ax=axes[1, 1], cmap="gray_r")

    ci, al, bl = bnl.mtr.make_common_itvls([a.itvls], [a.labels], [b.itvls], [b.labels])
    common_bs = mir_eval.util.intervals_to_boundaries(ci)
    intersection = bnl.mtr._meet(al) * bnl.mtr._meet(bl)

    false_positives = bnl.mtr._meet(bl) - intersection
    false_negatives = bnl.mtr._meet(al) - intersection
    # bnl.viz.sq(intersection, common_bs, ax=axes[2, 0])
    bnl.viz.sq(
        -false_negatives.astype(int) + intersection.astype(int),
        common_bs,
        ax=axes[2, 0],
    )
    bnl.viz.sq(
        -false_positives.astype(int) + intersection.astype(int),
        common_bs,
        ax=axes[2, 1],
    )

    axes[0, 0].set(title="Reference Segmentation", xticks=[], xlabel="")
    axes[0, 1].set(title="Estimated Segmentation", xticks=[], xlabel="")
    axes[1, 0].set(title="Label Agreement Map $M$", xticks=[], xlabel="")
    axes[1, 1].set(title="Label Agreement Map $M$", xticks=[], xlabel="", ylabel="")
    axes[2, 0].set(title="Intersection and Difference")
    axes[2, 1].set(title="Intersection and Difference", ylabel="")

    fig.savefig("./explain_pfc.pdf", bbox_inches="tight", transparent=True)


def frame_size_deviation():
    """Generate the frame size metrics comparison plot."""
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns

    results = xr.open_dataarray("./scripts/new_faster_compare.nc")
    metrics_list = ["pairwise", "vmeasure", "lmeasure"]

    fig, axes = plt.subplots(1, 3, figsize=(8, 1.8), sharey=True, sharex=True)

    for ax, metric in zip(axes, metrics_list):
        my_results = results.sel(output="f", metric=metric, frame_size=0)
        all_results = results.sel(output="f", metric=metric).drop_sel(frame_size=0)
        residual = all_results - my_results
        residual.name = "Residual"

        residual_df = (
            residual.to_dataframe().reset_index().drop(columns=["output", "metric"])
        )

        sns.stripplot(
            data=residual_df,
            x="Residual",
            y="frame_size",
            color="maroon",
            alpha=0.1,
            jitter=0.1,
            size=2,
            ax=ax,
            orient="h",
            rasterized=True,
        )
        sns.boxplot(
            data=residual_df,
            x="Residual",
            y="frame_size",
            orient="h",
            fill=False,
            fliersize=0,
            ax=ax,
        )
        ax.axvline(0, zorder=-1, color="k", linewidth=2)
        ax.set_xscale("symlog", linthresh=0.001)
        ax.grid(True)
        ax.set_title(metric.capitalize())

    axes[0].set(ylabel="Frame Size (seconds)", xlabel="")
    axes[1].set(xlabel="")
    axes[2].set(xlabel="", title="L-measure")
    plt.tight_layout()
    plt.savefig("./frame_size_metrics_comparison.pdf", bbox_inches="tight")


def depth_sweep():
    da = xr.open_dataarray("./depth_sweep.nc")
    my_result = da.sel(output="run_time", version="my")
    my_result.name = "my_run_time"
    me_result = da.sel(output="run_time", version="mir_eval")
    me_result.name = "mir_eval_run_time"

    runtime_da = da.sel(output="run_time")

    # Convert runtime_da to DataFrame
    df = runtime_da.to_dataframe(name="run_time").reset_index()
    # Offset levels by 1
    df["level"] += 1

    plt.figure(figsize=(4, 3))

    ax = sns.lineplot(
        x="level",
        y="run_time",
        hue="version",
        data=df,
        markers="o",
        dashes=False,
        errorbar=("ci", 99.9),
        palette="deep",
    )

    plt.yscale("log")
    plt.xlabel("Hierarchy Depth")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Distribution vs Number of Levels")
    plt.grid(True)
    plt.xlim(1, 12)

    # Modify legend text to "mir_eval" and "ours"
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        "Frame size = 0.1s" if label == "mir_eval" else "Continuous" for label in labels
    ]
    plt.legend(handles, new_labels)
    plt.tight_layout()
    plt.savefig("./depth_sweep_runtime.pdf", bbox_inches="tight", transparent=True)


def runtime_sweep():
    """Generate the runtime vs duration plot for different metrics."""
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import json
    import numpy as np
    from matplotlib.ticker import LogLocator, LogFormatter

    def plot_runtime(ax, metric="pairwise", **kwargs):
        """Plot runtime vs duration for a specific metric."""
        results = xr.open_dataarray("./new_faster_compare.nc")
        all_results = results.sel(output="run_time", metric=metric)
        all_results.name = "run_time"
        run_time_df = (
            all_results.to_dataframe().reset_index().drop(columns=["output", "metric"])
        )

        # Custom palette: 5 shades from black to gray, plus a bright red
        # add 0, 0 to DF so it all connects down
        run_time_df = pd.concat(
            [run_time_df, pd.DataFrame([{"tid": 0, "run_time": 0, "duration": 0}])],
            ignore_index=True,
        )
        custom_palette = ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"]
        cp = ["#FF0000"] + custom_palette
        dur_dict = json.load(open("salami_durations.json"))
        run_time_df["duration"] = run_time_df["tid"].map(dur_dict)
        run_time_df["duration"] = np.round((run_time_df["duration"] / 15)) * 15 + 0.1

        # Plot run_time_df using seaborn lineplot on the provided axes (ax)
        sns.lineplot(
            data=run_time_df,
            x="duration",
            y="run_time",
            hue="frame_size",
            alpha=0.5,
            palette=cp,
            ax=ax,
            **kwargs,
        )

        ax.set_yscale("log")
        return ax

    # Create the figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True, sharex=True)

    plot_runtime(axes[1], "vmeasure")

    # Set titles and labels
    axes[0].set_title("Pairwise")
    axes[1].set_title("V-measure")
    axes[2].set_title("L-measure")
    plot_runtime(axes[2], "lmeasure", legend=False)
    plot_runtime(axes[0], "pairwise", legend=False)
    axes[0].set_ylabel("Runtime (seconds)")
    axes[1].set_xlabel("Duration (seconds)")
    axes[2].set_xlabel("")
    axes[0].set_xlabel("")

    # Add clean gridlines and customize y-ticks for all subplots
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)  # Put grid behind the data

        # Customize y-ticks to show fewer, cleaner values
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))

        # Make minor gridlines lighter
        ax.grid(True, which="minor", alpha=0.1, linestyle="-", linewidth=0.3)

    # Adjust legend labels: replace '0.0' with 'event-based'
    handles, labels = axes[1].get_legend_handles_labels()
    new_labels = ["event-based" if label == "0.0" else label + "s" for label in labels]
    axes[1].legend(handles, new_labels, title="Frame Size", fontsize=8.5, ncol=2)

    # Save the plot with clean styling
    fig.tight_layout()
    plt.savefig("./runtime_vs_duration_br2.pdf", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python plots.py <command>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "explain_triplet":
        explain_triplet()
    elif command == "frame_size_deviation":
        frame_size_deviation()
    elif command == "explain_pfc":
        explain_pfc()
    elif command == "depth_sweep":
        depth_sweep()
    elif command == "runtime_sweep":
        runtime_sweep()
    else:
        raise NotImplementedError(f"Command '{command}' is not implemented")
