"""Visualization utilities for segmentations."""

import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from cycler import cycler

# Import Segmentation for type hinting only to avoid circular dependency
if TYPE_CHECKING:
    from .core import Segmentation  # pragma: no cover


def label_style_dict(labels, boundary_color="white", **kwargs):
    """
    Creates a mapping of labels to matplotlib style properties.

    Parameters
    ----------
    labels : list or ndarray
        List of labels (can be nested). Duplicates are processed once.
    boundary_color : str, default="white"
        Color for segment boundaries (edgecolor).
    **kwargs : dict
        Additional style properties to apply to all labels.

    Returns
    -------
    dict
        {label: {style_property: value}} mapping with keys like 'facecolor',
        'edgecolor', 'linewidth', 'hatch', and 'label'.
    """
    # Extract unique labels from potentially nested structure
    unique_labels = np.unique(
        np.concatenate([np.atleast_1d(l) for l in labels if l is not None])
    )

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

    # Create style mapping for each unique label
    seg_map = {}
    for lab, properties in zip(unique_labels, p_cycler):
        # Extract relevant style properties
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }

        # Convert color to facecolor to preserve edgecolor on rectangles
        if "color" in style:
            style["facecolor"] = style.pop("color")

        # Build final style dictionary
        seg_map[lab] = {
            "linewidth": 1,
            "edgecolor": boundary_color,
            "label": lab,
            **style,
            **kwargs,
        }
    return seg_map


def plot_segment(
    seg: "Segmentation",
    ax: Optional[plt.Axes] = None,
    label_text: bool = True,
    title: bool = True,
    ytick: str = "",
    time_ticks: bool = True,
    style_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a Segmentation object.

    Parameters
    ----------
    seg : bnl.core.Segmentation
        The Segmentation object to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    label_text : bool, default=True
        Whether to display segment labels as text on the plot.
    title : bool, default=True
        Whether to display a title on the axis.
    ytick : str, default=""
        Label for the y-axis. If empty, no label is shown.
    time_ticks : bool, default=True
        Whether to display time ticks and labels on the x-axis.
    style_map : dict, optional
        A precomputed mapping from labels to style properties.
        If None, it will be generated using `label_style_dict`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 0.6))  # short and wide
    else:
        fig = ax.figure

    # only plot if there are segments
    if len(seg) > 0:
        # Generate style map if not provided
        if style_map is None:
            style_map = label_style_dict(seg.labels)

        ax.set_xlim(seg.start, seg.end)
        for span in seg.segments:
            # Plot the segment using TimeSpan interface
            span.plot(ax=ax, text=label_text, **style_map[span.name])
    else:
        ax.text(
            0.5,
            0.5,
            "Empty Segmentation",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )

    if title and seg.name:
        ax.set_title(seg.name)

    # Set xlim only if start and end are different to avoid matplotlib warning
    if seg.start != seg.end:
        ax.set_xlim(seg.start, seg.end)
    else:
        # For empty or zero-duration segments, set a small default range
        ax.set_xlim(-0.1, 0.1)

    ax.set_ylim(0, 1)

    if time_ticks:
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xticks([])

    if ytick:
        ax.set_yticks([0.5])
        ax.set_yticklabels([ytick])
    else:
        ax.set_yticks([])

    return fig, ax
