"""Visualization tools for bnl."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from bnl.core import MultiSegment, Segment, TimeSpan


def create_style_map(
    labels: set[str],
    colormap: str = "tab20b",
) -> dict[str, dict[str, Any]]:
    """
    Creates a default, consistent color map for all labels.

    Parameters
    ----------
    labels : set[str]
        A set of unique label names.
    colormap : str, optional
        The matplotlib colormap to use, by default "tab20b".

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary mapping each label to its style properties.
    """
    unique_labels = sorted(list(labels))
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    style_map = {}

    for label, color in zip(unique_labels, colors, strict=False):
        r, g, b, _ = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        textcolor = "white" if luminance < 0.5 else "black"
        style = {"color": color, "textcolor": textcolor}
        style_map[label] = style
    return style_map


def plot_timespan(
    span: TimeSpan,
    ax: Axes,
    style: dict[str, Any] | None = None,
    y_pos: float = 0.0,
    height: float = 1.0,
    **kwargs: Any,
) -> Axes:
    """
    Plots a time span as a labeled rectangle on a set of axes.
    """
    style = style or {}
    color = style.get("color", "gray")
    textcolor = style.get("textcolor", "white")

    ax.axvspan(
        span.start.time,
        span.end.time,
        ymin=y_pos,
        ymax=y_pos + height,
        facecolor=color,
        edgecolor="white",
        linewidth=0.5,
        **kwargs,
    )

    center_time = span.start.time + span.duration / 2
    ax.text(
        center_time,
        y_pos + height / 2,
        span.name,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=8,
    )
    return ax


def plot_segment(
    segment: Segment,
    ax: Axes | None = None,
    style_map: dict[str, dict[str, Any]] | None = None,
    y_pos: float = 0.0,
    height: float = 1.0,
) -> Axes:
    """
    Plots a Segment by drawing each of its sections.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(15, len(segment.sections) * 0.5))
        ax.set(title=segment.name, yticks=[], xlim=(segment.start.time, segment.end.time), xlabel="Time (s)")

    if style_map is None:
        style_map = create_style_map(set(segment.labels))

    for section in segment.sections:
        # using str() to guard sections with empty string as name, so it's not just gray
        section_style = style_map.get(str(section), {})
        plot_timespan(section, ax, style=section_style, y_pos=y_pos, height=height)
    return ax


def plot_multisegment(
    ms: MultiSegment,
    ax: Axes | None = None,
    style_map: dict[str, dict[str, Any]] | None = None,
    colormap: str = "tab20b",
) -> Axes:
    """
    Plots all layers of the MultiSegment.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(15, len(ms.layers) * 0.5))
        ax.set(title=ms.name, yticks=[], xlim=(ms.start.time, ms.end.time), xlabel="Time (s)")

    if style_map is None:
        unique_labels = {label for layer in ms.layers for label in layer.labels}
        style_map = create_style_map(unique_labels, colormap)
    num_layers = len(ms.layers)
    for i, layer in enumerate(ms.layers):
        y_pos = 1 - (i + 1) / num_layers
        height = 1 / num_layers
        plot_segment(layer, ax, style_map=style_map, y_pos=y_pos, height=height)

    ax.set_yticks([1 - (i + 0.5) / num_layers for i in range(num_layers)])
    ax.set_yticklabels([layer.name for layer in ms.layers])

    return ax
