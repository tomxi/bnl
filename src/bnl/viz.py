"""Interactive visualization tools for bnl using Plotly."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import plotly.colors as pc
import plotly.graph_objects as go

if TYPE_CHECKING:
    from bnl.core import BoundaryContour, MultiSegment, Segment


__all__ = [
    "create_style_map",
    "plot_multisegment",
    "plot_boundary_contour",
]


def create_style_map(
    labels: list[str],
    colorscale: str | list[str] = "D3",
    patterns: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Creates color, pattern, and text style maps for a list of labels."""
    if isinstance(colorscale, str):
        colorscale = getattr(pc.qualitative, colorscale)

    if not patterns:
        patterns = ["", ".", "x", "/", "+", "|", "-", "\\"]
    patterns = patterns * (len(labels) // len(patterns) + 1)

    segment_bar_style = {
        label: dict(
            color=colorscale[i % len(colorscale)],
            pattern_shape=patterns[i // len(colorscale)],
            pattern_solidity=0.25,
            pattern_fgopacity=0.5,
            pattern_fgcolor="white",
            line=dict(width=0.5, color="white"),
        )
        for i, label in enumerate(labels)
    }
    return segment_bar_style


def _plot_bars_for_label(
    ms: MultiSegment | list[Segment],
    segment_bar_style: dict[str, dict[str, Any]],
) -> list[go.Bar]:
    """Helper to create bar traces for all labels in a single pass through the data."""
    from collections import defaultdict

    # Single pass through all sections, group by label
    label_data: defaultdict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "durations": [],
            "start_times": [],
            "y_positions": [],
            "hover_texts": [],
            "text_labels": [],
        }
    )

    for layer in ms:
        for section in layer:
            data = label_data[str(section)]
            data["durations"].append(section.duration)
            data["start_times"].append(section.start.time)
            data["y_positions"].append(str(layer))
            data["text_labels"].append(str(section))
            data["hover_texts"].append(
                f"<b>{section}</b><br>"
                f"Layer: {layer}<br>"
                f"Start: {section.start.time:.3f}s<br>"
                f"End: {section.end.time:.3f}s<br>"
                f"Duration: {section.duration:.3f}s"
            )

    # Create traces for each label
    traces = []
    for label, data in label_data.items():
        if label in segment_bar_style:  # Only create traces for labels we have styles for
            traces.append(
                go.Bar(
                    name=label,
                    y=data["y_positions"],
                    x=data["durations"],
                    base=data["start_times"],
                    orientation="h",
                    width=1.0,
                    showlegend=True,
                    text=data["text_labels"],
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=data["hover_texts"],
                    marker=segment_bar_style[label],
                )
            )
        else:
            warnings.warn(f"Label {label} not found in segment_bar_style", stacklevel=2)

    return traces


def plot_multisegment(
    ms: MultiSegment,
    colorscale: str | list[str] = "D3",
    hatch: bool = True,
) -> go.Figure:
    """Plots all layers of the MultiSegment with interactive features.

    Args:
        ms (MultiSegment): The MultiSegment to plot.
        colorscale (str | list[str], optional): Plotly colorscale to use. Can be a
            qualitative scale name (e.g., "Set3", "Pastel") or a list of colors. Defaults to "D3".
        hatch (bool, optional): Whether to use hatch patterns for different
            labels. Defaults to True.

    Returns:
        Figure: The Plotly figure with the MultiSegment.
    """
    fig = go.Figure()
    fig.update_layout(
        title_text=ms.name,
        title_x=0.5,
        xaxis_title="Time (s)",
        yaxis_title=None,
        width=650,
        height=len(ms) * 25 + 70,
        showlegend=False,
        barmode="overlay",
        yaxis=dict(
            categoryorder="array",
            categoryarray=[layer.name for layer in reversed(ms)],
        ),
        xaxis=dict(range=[ms.start.time, ms.end.time]),
        margin=dict(l=20, r=20, t=40, b=20),  # make plot bigger
    )

    ordered_unique_labels = []
    seen_labels = set()
    for layer in ms:
        for section in layer:
            if section.name not in seen_labels:
                ordered_unique_labels.append(section.name)
                seen_labels.add(section.name)

    segment_bar_style = create_style_map(
        [label for label in ordered_unique_labels if label is not None],
        colorscale,
        patterns=None if hatch else [""],
    )

    # Plot the actual data as bar traces, which will also create the legend
    traces = _plot_bars_for_label(ms, segment_bar_style)
    for trace in traces:
        fig.add_trace(trace)

    return fig


def plot_boundary_contour(
    bc: BoundaryContour,
    line_color: str = "#666",  # a medium grey that's okay on both white and black backgrounds
) -> go.Figure:
    """Plots a BoundaryContour with interactive hover information.

    Args:
        bc (BoundaryContour): The BoundaryContour to plot.
        line_color (str, optional): Color of the boundary lines. Defaults to "black".

    Returns:
        Figure: The Plotly figure with the BoundaryContour.
    """
    fig = go.Figure()
    fig.update_layout(
        title_text=bc.name,
        title_x=0.5,
        width=650,
        height=300,
        xaxis_title="Time (s)",
        yaxis_title="Salience",
        xaxis=dict(range=[bc.start.time, bc.end.time]),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),  # make plot bigger
    )

    # Always add the baseline
    fig.add_hline(y=0, line_color=line_color, line_width=1, opacity=0.8)

    boundaries = bc.bs[1:-1]
    if boundaries:
        times = [b.time for b in boundaries]
        saliences = [b.salience for b in boundaries]

        # This is the idiomatic way to draw many disconnected lines (stems) in Plotly.
        # By creating a single trace with `None` separating the coordinates for each
        # line, we can draw all stems in a single, efficient batch operation.
        stem_x = [v for t in times for v in (t, t, None)]
        stem_y = [v for s in saliences for v in (0, s, None)]

        # Draw all stems in a single, efficient trace
        fig.add_trace(
            go.Scatter(
                x=stem_x,
                y=stem_y,
                mode="lines",
                line=dict(color=line_color, width=1),
                hovertemplate=(
                    "<b>Boundary</b><br>Time: %{x:.3f}s<br>Salience: %{y:.3f}<extra></extra>"
                ),
            )
        )

    return fig
