"""
Clean plotting script for paper figures.
Simplified version focusing on the 5 working plots from the original script.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from mir_eval import hierarchy, util
from bnl import H, fio, mtr, viz


class FigurePlotter:
    """Simple plotter for paper figures"""

    def __init__(self, output_dir="figs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load data once
        self.salami_durations = json.load(open("scripts/salami_durations.json"))
        self.depth_sweep_data = xr.open_dataarray("scripts/depth_sweep.nc")
        self.comparison_data = xr.open_dataarray("scripts/new_faster_compare.nc")

    def _save_plot(self, fig, filename):
        """Save figure with consistent settings"""
        path = self.output_dir / f"{filename}.pdf"
        fig.savefig(path, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def _apply_common_styles(self, ax, title, xlabel=None, ylabel=None):
        """Apply consistent styling to axes"""
        ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    def _make_hierarchies(self):
        """Create example hierarchies for triplet explanation"""
        # Hierarchy 1
        itvls1 = np.array([[0, 2.5], [2.5, 6.01]]) / 6.01 * 5.01
        labels1 = ["A", "B"]
        itvls2 = np.array([[0, 1.2], [1.2, 2.5], [2.5, 3.5], [3.5, 6.01]]) / 6.01 * 5.01
        labels2 = ["a", "b", "c", "b"]
        hier1 = H([itvls1, itvls2], [labels1, labels2])

        # Hierarchy 2
        itvls3 = np.array([[0, 1.2], [1.2, 4], [4, 6.01]]) / 6.01 * 5.01
        labels3 = ["Mi", "Re", "Do"]
        itvls4 = np.array([[0, 1.2], [1.2, 3], [3, 4], [4, 6.01]]) / 6.01 * 5.01
        labels4 = ["T", "PD", "D", "T"]
        itvls5 = (
            np.array(
                [[0, 1.2], [1.2, 2], [2, 3], [3, 4], [4, 4.7], [4.7, 5.3], [5.3, 6.01]]
            )
            / 6.01
            * 5.01
        )
        labels5 = ["I", "IV", "ii", "V", "I", "IV", "I"]
        hier2 = H([itvls3, itvls4, itvls5], [labels3, labels4, labels5])

        hier1.update_sr(25)
        hier2.update_sr(25)
        return dict(h1=hier1, h2=hier2)

    def _plot_column(self, hier, q_fraq=0.3, axs=None):
        """Plot hierarchy analysis column"""
        ts = (hier.ticks[:-1] + hier.ticks[1:]) / 2
        q = int(q_fraq * (len(ts) - 1))
        axs = np.asarray(axs).flatten()

        num_rows = hier.d + 3
        empty_rows = len(axs) - num_rows
        y = 1.0
        for i in range(empty_rows):
            axs[i].axis("off")
            y -= 1
        hier.plot(
            axs=axs[empty_rows : empty_rows + hier.d], text=False, time_ticks=False
        )
        axs[0].set_title("Hierarchy", y=y)

        next_row = empty_rows + hier.d
        meet = hierarchy._meet(
            hier.itvls, hier.labels, frame_size=1.0 / hier.sr
        ).toarray()
        viz.sq(meet, hier.ticks, ax=axs[next_row], cmap="gray_r")
        axs[next_row].set_title("Label Agreement Map $M$")
        axs[next_row].vlines(
            ts[q], ts[0], ts[-1], colors="r", linestyles="dashed", label="Query time"
        )
        axs[next_row].hlines(ts[q], ts[0], ts[-1], colors="r", linestyles="dashed")
        axs[next_row].legend()

        axs[next_row + 1].sharex(axs[next_row])
        q_rel = meet[q]
        axs[next_row + 1].plot(ts, q_rel, color="k")
        axs[next_row + 1].fill_between(ts, q_rel, color="pink", alpha=0.8)
        axs[next_row + 1].axvline(ts[q], color="r", linestyle="dashed")
        axs[next_row + 1].set(
            title="Relevance to t: $M(t,\\cdot)$",
            xlabel="Time (s)",
            ylabel="Meet Depth",
        )

        u_more_relevant_mat = np.less.outer(q_rel, q_rel)
        viz.sq(u_more_relevant_mat, hier.ticks, ax=axs[next_row + 2], cmap="Reds")
        axs[next_row + 2].set(
            title="Relevant Triplets $\mathcal{A}(H;t)$",
            xlabel="u (sec)",
            ylabel="v (sec)",
        )
        return axs[0].get_figure(), axs, u_more_relevant_mat

    def _style_triplet_axes(self, axs):
        for ax in axs[-3:, :].flatten():
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linestyle("dashed")
                spine.set_linewidth(1.5)

        for ax in axs.flat[-8:]:
            ax.label_outer()
        for ax in axs.flat[-6:]:
            xticks = ax.get_xticks()
            ax.set_xticks(xticks[::2])
        for ax in axs.flat[-8:-6]:
            yticks = ax.get_yticks()
            ax.set_yticks(yticks[::2])
        for ax in axs.flat[-4:]:
            yticks = ax.get_yticks()
            ax.set_yticks(yticks[::2])

    def plot_explain_triplet(self):
        """Generate the triplet explanation plot."""
        hiers = self._make_hierarchies()
        h1 = hiers["h1"]
        h2 = hiers["h2"]
        max_d = max(h1.d, h2.d)
        fig, axs = viz.create_fig(
            w_ratios=[1, 1],
            h_ratios=[0.5] * max_d + [10, 2.5, 10, 10],
            figsize=(5, 9),
            h_gaps=[0.001] * (max_d - 1) + [0.001] * 4,
        )

        fig, _, r_sig_pairs = self._plot_column(h1, axs=axs[:-1, 0])
        fig, _, e_sig_pairs = self._plot_column(h2, axs=axs[:-1, 1])

        y_max_left = max(axs[-3, 0].get_ylim())
        y_max_right = max(axs[-3, 1].get_ylim())
        y_max = max(y_max_left, y_max_right)

        y_ticks = np.arange(0, int(np.ceil(y_max)))
        axs[-3, 0].set_yticks(y_ticks)
        axs[-3, 1].set_yticks(y_ticks)

        intersect_pairs = (r_sig_pairs * e_sig_pairs).astype(int)
        false_negative_pairs = -intersect_pairs + r_sig_pairs.astype(int)
        false_positive_pairs = -intersect_pairs + e_sig_pairs.astype(int)

        self._apply_common_styles(axs[0, 0], "Reference Hierarchy $H$")
        self._apply_common_styles(axs[0, 1], "Estimated Hierarchy $\hat{H}$")

        filtered_ref = intersect_pairs - false_negative_pairs
        viz.sq(filtered_ref, h1.ticks, ax=axs[-1, 0])
        self._apply_common_styles(
            axs[-1, 0], "Intersection", xlabel="u (sec)", ylabel="v (sec)"
        )

        filtered_est = intersect_pairs - false_positive_pairs
        viz.sq(filtered_est, h2.ticks, ax=axs[-1, 1])
        self._apply_common_styles(
            axs[-1, 1], "Intersection", xlabel="u (sec)", ylabel="v (sec)"
        )
        self._apply_common_styles(
            axs[-2, -1],
            "Relevant Triplets $\mathcal{A}(\hat{H};t)$",
        )

        self._style_triplet_axes(axs)

        self._save_plot(fig, "explain_triplet")

    def plot_explain_pfc(self):
        """Generate the pairwise false positive/negative explanation plot."""
        hiers = list(fio.salami_ref_hiers(464).values())
        a = hiers[0].levels[0]
        b = hiers[1].levels[0]

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

        # Plot agreement matrices
        viz.sq(a.A(), a.beta, ax=axes[1, 0], cmap="gray_r")
        viz.sq(b.A(), b.beta, ax=axes[1, 1], cmap="gray_r")

        # Compute common intervals and intersections
        ci, al, bl = mtr.make_common_itvls([a.itvls], [a.labels], [b.itvls], [b.labels])
        common_bs = util.intervals_to_boundaries(ci)
        intersection = mtr._meet(al) * mtr._meet(bl)

        false_positives = mtr._meet(bl) - intersection
        false_negatives = mtr._meet(al) - intersection
        viz.sq(
            -false_negatives.astype(int) + intersection.astype(int),
            common_bs,
            ax=axes[2, 0],
        )
        viz.sq(
            -false_positives.astype(int) + intersection.astype(int),
            common_bs,
            ax=axes[2, 1],
        )

        # Style axes
        self._apply_common_styles(axes[0, 0], "Reference Segmentation")
        self._apply_common_styles(axes[0, 1], "Estimated Segmentation")
        self._apply_common_styles(axes[1, 0], "Label Agreement Map $M$")
        self._apply_common_styles(axes[1, 1], "Label Agreement Map $M$")
        self._apply_common_styles(axes[2, 0], "Intersection")
        self._apply_common_styles(axes[2, 1], "Intersection")

        for ax in axes.flat:
            ax.label_outer()

        self._save_plot(fig, "explain_pfc")

    def plot_frame_size_deviation(self):
        """Generate frame size metrics comparison"""
        results = self.comparison_data
        metrics_list = ["pairwise", "vmeasure", "lmeasure"]
        titles = ["Pairwise", "V-measure", "L-measure"]
        fig, axes = plt.subplots(1, 3, figsize=(8, 1.9), sharey=True, sharex=True)

        for ax, metric, title in zip(axes, metrics_list, titles):
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
            self._apply_common_styles(ax, title)
            xticks = ax.get_xticks()
            ax.set_xticks(xticks[::2])

        axes[0].set_ylabel("Frame Size (sec)")
        axes[1].set_xlabel("Metric Deviation")
        axes[0].set_xlabel("")
        axes[2].set_xlabel("")
        fig.tight_layout()
        self._save_plot(fig, "frame_size_metrics_comparison")

    def plot_depth_sweep(self):
        """Generate the depth sweep runtime plot."""
        da = self.depth_sweep_data
        df = da.sel(output="run_time").to_dataframe(name="run_time").reset_index()
        df["level"] += 1

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.lineplot(
            x="level",
            y="run_time",
            hue="version",
            data=df,
            markers="o",
            dashes=False,
            errorbar=("ci", 95),
            palette="deep",
            ax=ax,
        )
        ax.set_yscale("log")
        self._apply_common_styles(
            ax,
            "Effect of Hierarchy Depth\n on L-measure Runtime",
            "Depth of Estimated Hierarchy",
            "Runtime (sec)",
        )
        ax.set_xlim(1, 12)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [
            "frame size = 0.1s" if label == "mir_eval" else "event-based"
            for label in labels
        ]
        ax.legend(handles, new_labels)
        fig.tight_layout()
        self._save_plot(fig, "depth_sweep_runtime")

    def plot_runtime_sweep(self):
        """Generate the runtime vs duration plot for different metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(8, 2.3), sharey=True, sharex=True)
        metrics = ["pairwise", "vmeasure", "lmeasure"]
        titles = ["Pairwise", "V-measure", "L-measure"]

        custom_palette = ["#FF0000"] + [
            "#000000",
            "#333333",
            "#666666",
            "#999999",
            "#CCCCCC",
        ]

        for ax, metric, title in zip(axes, metrics, titles):
            # Get data for this metric
            results = self.comparison_data
            all_results = results.sel(output="run_time", metric=metric)
            all_results.name = "run_time"
            run_time_df = (
                all_results.to_dataframe()
                .reset_index()
                .drop(columns=["output", "metric"])
            )
            run_time_df = pd.concat(
                [run_time_df, pd.DataFrame([{"tid": 0, "run_time": 0, "duration": 0}])],
                ignore_index=True,
            )

            # Map durations and round
            run_time_df["duration"] = run_time_df["tid"].map(self.salami_durations)
            run_time_df["duration"] = (
                np.round((run_time_df["duration"] / 15)) * 15 + 0.1
            )

            # Create the line plot
            sns.lineplot(
                data=run_time_df,
                x="duration",
                y="run_time",
                hue="frame_size",
                alpha=0.5,
                palette=custom_palette,
                ax=ax,
                legend=(metric == "vmeasure"),
                errorbar=("ci", 95),
            )

            # Apply styling
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1e2)
            self._apply_common_styles(ax, title)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"$10^{{{int(np.log10(x))}}}$")
            )
            ax.grid(True, which="minor", alpha=0.1, linestyle="-", linewidth=0.3)

            # Handle legend for vmeasure subplot
            if metric == "vmeasure":
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [
                    "event-based" if label == "0.0" else label + "s" for label in labels
                ]
                ax.legend(handles, new_labels, title="Frame Size", fontsize=8.5, ncol=2)

        axes[0].set_ylabel("Runtime (sec)")
        axes[1].set_xlabel("Track Duration (sec)")
        axes[0].set_xlabel("")
        axes[2].set_xlabel("")
        fig.tight_layout()
        self._save_plot(fig, "runtime_vs_duration_br")

    def plot_all(self):
        """Generate all figures"""
        self.plot_explain_triplet()
        self.plot_frame_size_deviation()
        self.plot_explain_pfc()
        self.plot_depth_sweep()
        self.plot_runtime_sweep()


def main():
    """Main function for command-line usage

    This script reproduces and makes the figures in the "Lose the Frame" paper.
    """
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Lose the Frame Figure Generator")
        print(
            "This script reproduces and makes the figures in the 'Lose the Frame' paper."
        )
        print()
        print("Usage: python lose_the_frame_plots.py <command> [output_dir]")
        print(
            "Commands: explain_triplet, frame_size_deviation, explain_pfc, depth_sweep, runtime_sweep, all"
        )
        print("output_dir: Optional relative directory for output (default: ./figs)")
        sys.exit(1)

    command = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else "./figs"
    plotter = FigurePlotter(output_dir=output_dir)

    plot_methods = {
        "explain_triplet": plotter.plot_explain_triplet,
        "frame_size_deviation": plotter.plot_frame_size_deviation,
        "explain_pfc": plotter.plot_explain_pfc,
        "depth_sweep": plotter.plot_depth_sweep,
        "runtime_sweep": plotter.plot_runtime_sweep,
        "all": plotter.plot_all,
    }

    if command in plot_methods:
        plot_methods[command]()
        print(f"✓ Completed: {command}")
    else:
        print(f"✗ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
