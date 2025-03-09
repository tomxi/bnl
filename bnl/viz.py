import numpy as np
import pandas as pd
import librosa
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .formatting import multi2hier, hier2mireval
from .external import reindex

import itertools
from cycler import cycler


def scatter_scores(
    x_data: pd.Series,
    y_data: pd.Series,
    title: str = "L scores per track",
    xlabel: str = "x label",
    ylabel: str = "y label",
    ax: any = None,
) -> matplotlib.axes._axes.Axes:
    """
    Create a scatter plot of two data series.

    This function plots two data series as a scatter plot, with optional customizations for title, axis labels, and using an existing axes object.
    It also adds a diagonal line from (0,0) to (1,1), sets equal aspect ratio, and limits both axes to the range [0, 1].

    Parameters
    ----------
    x_data : pd.Series
        Data for x-axis values
    y_data : pd.Series
        Data for y-axis values
    title : str, optional
        Plot title, defaults to "L scores per track"
    xlabel : str, optional
        X-axis label, defaults to "x label"
    ylabel : str, optional
        Y-axis label, defaults to "y label"
    ax : matplotlib.axes._axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure and axes.

    Returns
    -------
    matplotlib.axes._axes.Axes
        The axes object containing the scatter plot
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        x=x_data,
        y=y_data,
        alpha=0.5,
        s=3,
    )
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.plot([0, 1], [0, 1], "r:")
    ax.set_aspect("equal", "box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def assign_label_styles(labels, **kwargs):
    """
    Creates a mapping of labels to visual style properties for consistent visualization.

    This function processes a list of labels (which may contain nested lists),
    extracts unique labels, sorts them, and assigns consistent visual styles
    (like colors and hatch patterns) to each label.

    Parameters
    ----------
    labels : list
        List of labels (can contain nested lists). Duplicate labels will be handled only once.
    **kwargs : dict
        Additional style properties to apply to all labels. These will override
        the default styles if there are conflicts.

    Returns
    -------
    dict
        A dictionary mapping each unique label to its style properties.
        Each entry contains properties like 'facecolor', 'edgecolor', 'linewidth',
        'hatch', and 'label'.

    Notes
    -----
    - If there are 80 or fewer unique labels, uses 'tab10' colormap with 8 hatch patterns.
    - If there are more than 80 unique labels, uses 'tab20' colormap with 15 hatch patterns.
    - Default styles include white edgecolor and linewidth of 1.
    """
    labels = labels.copy()
    unique_labels = []
    while labels:
        l = labels.pop()
        if isinstance(l, list):
            labels.extend(l)
        elif l not in unique_labels:
            unique_labels.append(l)
    unique_labels.sort()

    if len(unique_labels) <= 80:
        hatch_cycler = cycler(hatch=["", "..", "xx", "O.", "*", "\\O", "oo", "xxO"])
        fc_cycler = cycler(color=plt.get_cmap("tab10").colors)
        p_cycler = hatch_cycler * fc_cycler
    else:
        hatch_cycler = cycler(
            hatch=[
                "",
                "oo",
                "xx",
                "O.",
                "*",
                "..",
                "\\",
                "\\O",
                "--",
                "oo--",
                "xx--",
                "O.--",
                "*--",
                "\\--",
                "\\O--",
            ]
        )
        fc_cycler = cycler(color=plt.get_cmap("tab20").colors)
        p_cycler = itertools.cycle(hatch_cycler * fc_cycler)

    seg_map = dict()
    for lab, properties in zip(unique_labels, p_cycler):
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }
        # Swap color -> facecolor here so we preserve edgecolor on rects
        if "color" in style:
            style.setdefault("facecolor", style["color"])
            style.pop("color", None)
        seg_map[lab] = dict(linewidth=1, edgecolor="white")
        seg_map[lab].update(style)
        seg_map[lab].update(kwargs)
        seg_map[lab]["label"] = lab
    return seg_map


def flat_segment(
    intervals,
    labels,
    ax=None,
    text=False,
    style_map=None,
):
    """Plot a single layer of flat segmentation."""
    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_yticks([])
    ax.set_xlim(intervals[0][0], intervals[-1][-1])

    if style_map is None:
        style_map = assign_label_styles(labels, edgecolor="white")
    transform = ax.get_xaxis_transform()

    for ival, lab in zip(intervals, labels):
        rect = ax.axvspan(ival[0], ival[1], ymin=0, ymax=1, **style_map[lab])
        if text:
            ann = ax.annotate(
                lab,
                xy=(ival[0], 1),
                xycoords=transform,
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                clip_on=True,
                bbox=dict(boxstyle="round", facecolor="white"),
            )
            ann.set_clip_path(rect)
    return ax


def multi_seg(
    ms_anno,
    figsize=(8, 4),
    relabel=True,
    legend_ncol=6,
    text=False,
    y_label=True,
    x_label=True,
    legend_offset=-0.06,
    axs=None,
):
    """Plots the given multi_seg annotation."""
    hier = multi2hier(ms_anno)
    if relabel:
        hier = reindex(hier)
    N = len(hier)
    if axs is None:
        fig, axs = plt.subplots(N, figsize=figsize)
    if N == 1 and not isinstance(axs, list):
        axs = [axs]

    _, lbls = hier2mireval(hier)
    style_map = assign_label_styles(lbls)
    legend_handles = [mpatches.Patch(**style) for style in style_map.values()]

    for level, (itvl, lbl) in enumerate(hier):
        ax = flat_segment(itvl, lbl, ax=axs[level], style_map=style_map, text=text)
        if y_label:
            ax.set_yticks([0.5])
            ax.set_yticklabels([level + 1])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    if x_label:
        # Show time axis on the last layer
        axs[-1].xaxis.set_major_locator(ticker.AutoLocator())
        axs[-1].xaxis.set_major_formatter(librosa.display.TimeFormatter())
        axs[-1].set_xlabel("Time")

    if legend_ncol:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=legend_ncol,
            bbox_to_anchor=(0.5, legend_offset),
        )
    if y_label:
        fig.text(0.94, 0.55, "Segmentation Levels", va="center", rotation="vertical")
    return fig, axs


def heatmap(
    da,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    colorbar=True,
    figsize=(5, 5),
    no_deci=False,
):
    """Plot a heatmap of a 2D DataArray."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    da = da.squeeze()
    if len(da.shape) == 1:
        da = da.expand_dims(dim="_", axis=0)
        da = da.assign_coords(_=[""])

    im = ax.imshow(da.values.astype(float), cmap="coolwarm")

    try:
        # try to get axis label and ticks from dataset coords
        ycoord, xcoord = da.dims
        xticks = da.indexes[xcoord]
        yticks = da.indexes[ycoord]
        if xlabel is None:
            xlabel = xcoord
        if ylabel is None:
            ylabel = ycoord

        ax.set_xticks(np.arange(len(xticks)), labels=xticks)
        ax.set_yticks(np.arange(len(yticks)), labels=yticks)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    except:
        pass

    for i in range(da.shape[0]):
        for j in range(da.shape[1]):
            if no_deci:
                ax.text(j, i, f"{da.values[i, j]}", ha="center", va="center", color="k")
            else:
                ax.text(
                    j, i, f"{da.values[i, j]:.3f}", ha="center", va="center", color="k"
                )

    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, shrink=0.8)
    return fig, ax
