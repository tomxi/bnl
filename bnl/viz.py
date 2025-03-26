# import librosa
from librosa.display import TimeFormatter, specshow
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import itertools
from cycler import cycler


def sq(mat, ticks, ax=None, **kwargs):
    """Plot a meet matrix for a given hierarchy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 3.5))
    quadmesh = specshow(
        mat,
        ax=ax,
        x_coords=ticks,
        y_coords=ticks,
        x_axis="time",
        y_axis="time",
        **kwargs,
    )
    return quadmesh


def label_style_dict(labels, boundary_color="white", **kwargs):
    """
    Creates a mapping of labels to visual style properties for consistent visualization.

    This function processes a list of labels (which may contain nested lists),
    extracts unique labels, sorts them, and assigns consistent visual styles
    (like colors and hatch patterns) to each label.

    Parameters
    ----------
    labels : nparray or list
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
    # flatten the labels list
    if type(labels) is list:
        unique_labels = np.unique(np.concatenate(labels))
    else:
        unique_labels = np.unique(labels)

    # More hatch patterns for more labels
    hatchs = ["", "..", "O.", "*", "xx", "xxO", "\\O", "oo", "\\"]
    more_hatchs = [h + "--" for h in hatchs]

    if len(unique_labels) <= 80:
        hatch_cycler = cycler(hatch=hatchs)
        fc_cycler = cycler(color=plt.get_cmap("tab10").colors)
        p_cycler = hatch_cycler * fc_cycler
    else:
        hatch_cycler = cycler(hatch=hatchs + more_hatchs)
        fc_cycler = cycler(color=plt.get_cmap("tab20").colors)
        # make it repeat...
        p_cycler = itertools.cycle(hatch_cycler * fc_cycler)

    # Create a mapping of labels to styles by cycling through the properties
    # and assigning them to the labels as they appear in the unique labels' ordering
    seg_map = dict()
    for lab, properties in zip(unique_labels, p_cycler):
        # set style according to p_cycler
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }
        # Swap color -> facecolor here so we preserve edgecolor on rects
        if "color" in style:
            style.setdefault("facecolor", style["color"])
            style.pop("color", None)
        seg_map[lab] = dict(linewidth=1, edgecolor=boundary_color)
        seg_map[lab].update(style)
        seg_map[lab].update(kwargs)
        seg_map[lab]["label"] = lab
    return seg_map


def segment(
    intervals,
    labels,
    ax,
    text=False,
    ytick="",
    time_ticks=False,
    style_map=None,
):
    """Plot a single layer of flat segmentation."""
    ax.set_xlim(intervals[0][0], intervals[-1][-1])

    if style_map is None:
        style_map = label_style_dict(labels, edgecolor="white")
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

    if time_ticks:

        # Use the default automatic tick locator
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(TimeFormatter())
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xticks([])

    if ytick == "":
        ax.set_yticks([])
    else:
        ax.set_yticks([0.5])
        ax.set_yticklabels([ytick])
    return ax.get_figure(), ax


def create_fig(
    h_ratios=[1, 1, 1],
    w_ratios=[1, 1],
    h_gaps=0.05,
    w_gaps=0.05,
    figsize=(5, 5),
):
    """
    Create a figure with specified dimensions and height/width ratios, with gaps between subplots.

    Parameters
    ----------
    figsize : tuple, optional
        Figure dimensions (width, height) in inches, by default (5, 5)
    h_gaps : float or list, optional
        Vertical gap size between subplots, by default 0.05
    w_gaps : float or list, optional
        Horizontal gap size between subplots, by default 0.05
    h_ratios : list, optional
        Relative heights of subplots, by default [1, 1, 1]
    w_ratios : list, optional
        Relative widths of subplots, by default [1, 1]

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    axs : 2D array
        2D array of subplot axes
    """
    # Create figure with tight layout
    fig = plt.figure(
        constrained_layout=True,
        figsize=figsize,
    )

    # Handle edge cases of single row/column
    if len(h_ratios) <= 0 or len(w_ratios) <= 0:
        return fig, []

    if len(h_ratios) == 1 and len(w_ratios) == 1:
        axs = [[fig.add_subplot(111)]]
        return fig, axs

    # Normalize h_gaps to a list with one fewer element than h_ratios
    if not isinstance(h_gaps, list):
        h_gaps = [h_gaps] * (len(h_ratios) - 1)
    while len(h_gaps) < len(h_ratios) - 1:
        h_gaps.append(h_gaps[-1])
    h_gaps = h_gaps[: len(h_ratios) - 1]

    # Normalize w_gaps to a list with one fewer element than w_ratios
    if not isinstance(w_gaps, list):
        w_gaps = [w_gaps] * (len(w_ratios) - 1)
    while len(w_gaps) < len(w_ratios) - 1:
        w_gaps.append(w_gaps[-1])
    normalized_w_gaps = w_gaps[: len(w_ratios) - 1]

    # Create interleaved heights using zip
    gridspec_heights = []
    for height, gap in zip(h_ratios[:-1], h_gaps):
        gridspec_heights.extend([height, gap])
    gridspec_heights.append(h_ratios[-1])  # Add final height without gap

    # Create interleaved widths using zip
    gridspec_widths = []
    for width, gap in zip(w_ratios[:-1], normalized_w_gaps):
        gridspec_widths.extend([width, gap])
    gridspec_widths.append(w_ratios[-1])  # Add final width without gap

    # Create gridspec
    gs = fig.add_gridspec(
        len(gridspec_heights),
        len(gridspec_widths),
        height_ratios=gridspec_heights,
        width_ratios=gridspec_widths,
    )

    # Create 2D array of axes
    axs = []
    for i in range(len(h_ratios)):
        row = []
        for j in range(len(w_ratios)):
            row.append(fig.add_subplot(gs[i * 2, j * 2]))
        axs.append(np.array(row))

    fig = row[0].get_figure()
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    return fig, np.array(axs)


# For paper figure on benchmarking continuous versus frame-based implementation.
def plot_scatter_frame_vs_continuous(
    results,
    sel_dict={"output": "run_time"},
    frame_sizes=[0.1, 0.25, 0.5, 1, 2],
    ax=None,
    color_start_idx=0,  # New parameter to specify starting color index
    **scatter_kwargs,
):
    """
    Generates a scatter plot comparing the performance of frame-based vs. continuous
    implementations for a given output type (e.g., 'output': 'run_time', 'lm', 'lr', 'lp).

    Args:
        results (xr.DataArray): An xarray DataArray containing the results,
            with dimensions 'tid', 'frame_size', and 'output'.
        output_type (str): The type of output to plot (e.g., 'run_time', 'lm').
        ax (matplotlib.axes._axes.Axes, optional): An existing matplotlib Axes
            object to plot on. If None, a new figure and axes will be created.
            Defaults to None.
        color_start_idx (int, optional): Index in the default color cycle to start from.
            Defaults to 0.

    Returns:
        matplotlib.axes._axes.Axes: The Axes object containing the plot.
    """
    # Name this selection
    name = "_".join(sel_dict.values())

    # Filter the xarray DataArray to only include the desired output type
    results_filtered = results.sel(sel_dict)
    results_filtered.name = name

    # Convert the filtered xarray DataArray to a pandas DataFrame
    df = results_filtered.to_dataframe().reset_index()
    # Group by 'tid' and 'frame_size' and calculate the mean for the specified output type
    df_grouped = df.groupby(["tid", "frame_size"])[name].mean().reset_index()
    # Pivot the DataFrame to have the specified output type for each 'frame_size' as columns
    df_pivot = df_grouped.pivot(index="tid", columns="frame_size", values=name)

    # Create the scatter plot if ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    # Get the current color cycle and rotate it to start at the specified index
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = list(prop_cycle.by_key()["color"])
    rotated_colors = colors[color_start_idx:] + colors[:color_start_idx]
    ax.set_prop_cycle(cycler(color=rotated_colors))

    kwargs = dict(alpha=0.35, s=3, edgecolor="none")
    kwargs.update(**scatter_kwargs)

    for frame_size in frame_sizes:
        if frame_size not in df_pivot.columns:
            raise ValueError(f"Frame size {frame_size} not found in the data.")

        ax.scatter(
            df_pivot[0.0],
            df_pivot[frame_size],
            label=f"{frame_size:.1f} sec",
            **kwargs,
        )

    ax.set_title(name)
    ax.grid(True)

    # Set the x and y axis limits, if max is over 2, then use log scale
    if df_pivot.max().max() > 2:
        ax.set_xscale("log")
        ax.set_yscale("log")
        min_v = df_pivot.min().min()
        max_v = df_pivot.max().max()
    else:
        min_v, max_v = 0, 1

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    # Set the aspect ratio to be equal
    ax.set_aspect("equal", adjustable="box")

    # Plot a diagonal line for reference
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="-",
        color="k",
        alpha=0.5,
        linewidth=0.5,
    )

    return ax
