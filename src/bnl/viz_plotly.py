"""Interactive visualization tools for bnl using Plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

if TYPE_CHECKING:
    from bnl.core import BoundaryContour, MultiSegment


def _get_contrasting_text_color(color_str: str) -> str:
    """Calculates whether black or white text is more readable on a given background color."""
    try:
        if color_str.startswith("#"):
            r, g, b = [int(color_str[i : i + 2], 16) for i in (1, 3, 5)]
        elif color_str.startswith("rgb"):
            parts = color_str.replace("rgb(", "").replace("rgba(", "").replace(")", "").split(",")
            r, g, b = [int(p.strip()) for p in parts[:3]]
        else:
            # Fallback for named colors, etc., though less common with Plotly scales
            from PIL import ImageColor

            r, g, b = ImageColor.getrgb(color_str)
    except (ValueError, ImportError):
        return "black"  # Default to black on parsing error

    # Using the W3C luminance algorithm
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "black"


def _create_style_maps(
    labels: list[str],
    colorscale: str = "Plotly",
    font_size: int = 10,
    all_hatch_patterns: list[str] | None = None,
    unique_hatch_color_diversity: int = 12,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    """Creates color, pattern, and text style maps for a list of labels."""
    try:
        colors = getattr(pc.qualitative, colorscale)
    except AttributeError:
        try:
            colors = pc.sample_colorscale(colorscale, np.linspace(0, 1, max(1, len(labels))))
        except (ValueError, KeyError):
            colors = pc.qualitative.Plotly

    if not all_hatch_patterns:
        all_hatch_patterns = ["", "/", "\\", "x", "-", "|", "+", "."]

    num_hatch_to_use = int(np.ceil(len(labels) / unique_hatch_color_diversity)) if len(labels) > 0 else 1
    patterns = all_hatch_patterns[: min(num_hatch_to_use, len(all_hatch_patterns))]

    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}
    pattern_map = {label: patterns[i % len(patterns)] for i, label in enumerate(labels)}
    text_style_map = {
        label: dict(color=_get_contrasting_text_color(color_map[label]), size=font_size) for label in labels
    }

    return color_map, pattern_map, text_style_map


def _plot_bars_for_label(
    fig: Figure,
    ms: MultiSegment,
    label: str,
    color_map: dict[str, str],
    pattern_map: dict[str, str],
    char_width_in_seconds: float,
    text_style_map: dict[str, dict],
):
    """Helper to plot all sections of a given label as a single horizontal bar trace."""
    durations, start_times, y_positions, hover_texts, annotations = [], [], [], [], []

    for layer in ms.layers:
        for section in layer.sections:
            if section.name == label:
                durations.append(section.duration)
                start_times.append(section.start.time)
                y_positions.append(layer.name)
                hover_texts.append(
                    f"<b>{section.name}</b><br>"
                    f"Layer: {layer.name}<br>"
                    f"Start: {section.start.time:.3f}s<br>"
                    f"End: {section.end.time:.3f}s<br>"
                    f"Duration: {section.duration:.3f}s"
                )
                text_to_display = section.name
                if len(section.name) * char_width_in_seconds > section.duration:
                    text_to_display = ""
                annotations.append(
                    dict(
                        x=section.start.time + section.duration / 2,
                        y=layer.name,
                        text=text_to_display,
                        showarrow=False,
                        font=text_style_map[label],
                    )
                )

    fig.add_trace(
        go.Bar(
            name=label,
            y=y_positions,
            x=durations,
            base=start_times,
            orientation="h",
            width=1.0,
            showlegend=True,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
            marker=dict(
                color=color_map[label],
                pattern_shape=pattern_map[label],
                pattern_solidity=0.8,
                line=dict(width=0.5, color="white"),
            ),
        )
    )
    # Batch-add annotations at the end for this trace
    for ann in annotations:
        fig.add_annotation(**ann)


def plot_multisegment(
    ms: MultiSegment,
    fig: Figure | None = None,
    figsize: tuple[float, float] | None = None,
    colorscale: str = "D3",
    hatch: bool = True,
) -> Figure:
    """
    Plots all layers of the MultiSegment with interactive features.

    Parameters
    ----------
    ms : MultiSegment
        The MultiSegment to plot.
    fig : Figure, optional
        Existing Plotly figure to add to. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in pixels.
    colorscale : str, default "Plotly"
        Plotly colorscale to use. Can be a qualitative scale name (e.g., "Set3", "Pastel")
        or a continuous scale to be sampled (e.g., "Viridis").

    Returns
    -------
    Figure
        The Plotly figure with the MultiSegment.
    """
    if fig is None:
        width, height = figsize or (800, len(ms.layers) * 25 + 200)
        fig = go.Figure()
        fig.update_layout(
            title_text=ms.name,
            xaxis_title="Time (s)",
            yaxis_title=None,
            width=width,
            height=height,
            showlegend=True,
            barmode="overlay",
            yaxis=dict(
                categoryorder="array",
                categoryarray=[layer.name for layer in reversed(ms.layers)],
            ),
            xaxis=dict(range=[ms.start.time, ms.end.time]),
        )

    # Heuristic to hide text that overflows its container
    total_duration = ms.end.time - ms.start.time
    pixels_per_second = width / total_duration if total_duration > 0 else 0
    font_size = 10
    # Estimate average character width as a factor of font size
    char_width_in_seconds = (font_size * 0.5) / pixels_per_second if pixels_per_second > 0 else float("inf")

    ordered_unique_labels = []
    seen_labels = set()
    for layer in ms.layers:
        for section in layer.sections:
            if section.name not in seen_labels:
                ordered_unique_labels.append(section.name)
                seen_labels.add(section.name)

    color_map, pattern_map, text_style_map = _create_style_maps(
        ordered_unique_labels, colorscale, font_size=font_size, all_hatch_patterns=None if hatch else []
    )

    # Plot the actual data as bar traces, which will also create the legend
    for label in ordered_unique_labels:
        _plot_bars_for_label(
            fig,
            ms,
            label,
            color_map,
            pattern_map,
            char_width_in_seconds,
            text_style_map,
        )

    return fig


def plot_boundary_contour(
    bc: BoundaryContour,
    fig: Figure | None = None,
    figsize: tuple[float, float] | None = None,
    marker_size: int = 8,
    line_color: str = "black",
    **kwargs: Any,
) -> Figure:
    """
    Plots a BoundaryContour with interactive hover information.

    Parameters
    ----------
    bc : BoundaryContour
        The BoundaryContour to plot.
    fig : Figure, optional
        Existing Plotly figure to add to. If None, creates a new figure.
    figsize : tuple, optional
        Figure size (width, height) in pixels.
    marker_size : int, default 8
        Size of boundary markers.
    line_color : str, default "black"
        Color of the boundary lines.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Figure
        The Plotly figure with the BoundaryContour.
    """
    if fig is None:
        width, height = figsize or (800, 400)
        fig = go.Figure()
        fig.update_layout(
            title=bc.name,
            xaxis_title="Time (s)",
            yaxis_title="Salience",
            xaxis=dict(range=[bc.start.time, bc.end.time]),
            width=width,
            height=height,
        )

    boundaries = bc.boundaries[1:-1]
    if boundaries:
        times = [b.time for b in boundaries]
        saliences = [b.salience for b in boundaries]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=saliences,
                mode="markers",
                marker=dict(size=marker_size, color=line_color),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    arrayminus=saliences,
                    color=line_color,
                    thickness=1.5,
                    width=0,
                ),
                hovertemplate=("<b>Boundary</b><br>Time: %{x:.3f}s<br>Salience: %{y:.3f}<extra></extra>"),
                showlegend=False,
            )
        )

    fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.5)

    return fig
