import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

import bnl
from bnl import ops


# --- Core Computational Class (Reused for both stages) ---
class KDEBoundaryExplorer:
    """
    Manages the state and computation for KDE-based boundary merging.
    This version is headless and optimized for use in a web app.
    """

    def __init__(self, data_points: np.ndarray, weights: np.ndarray = None, viz_resolution: int = 2000):
        if data_points.size == 0:
            # Return a non-functional but safe object if there's no data
            self.resampled_times = np.array([])
            self.viz_grid = np.array([])
            self.density = np.array([])
            self.peaks = np.array([])
            self.peak_saliences = np.array([])
            return

        # --- Resample ONCE based on weights ---
        if weights is not None and weights.sum() > 0:
            probabilities = weights / np.sum(weights)
            resample_size = min(len(data_points) * 10, 5000)
            self.resampled_times = np.random.choice(data_points, size=resample_size, p=probabilities).reshape(-1, 1)
        else:  # Handle unweighted case
            self.resampled_times = data_points.reshape(-1, 1)

        # --- Setup for visualization grid ---
        min_t, max_t = data_points.min(), data_points.max()
        # Add a small buffer to avoid plotting issues at the edges
        buffer = (max_t - min_t) * 0.05 if (max_t - min_t) > 0 else 0.1
        self.viz_grid = np.linspace(min_t - buffer, max_t + buffer, viz_resolution).reshape(-1, 1)

        self.density = np.zeros(self.viz_grid.shape)
        self.peaks = np.array([])
        self.peak_saliences = np.array([])

    def update_bandwidth(self, bandwidth: float):
        """
        Updates the KDE and finds new peaks for the given bandwidth.
        """
        if bandwidth <= 0 or self.resampled_times.size == 0:
            return

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(self.resampled_times)
        log_density = kde.score_samples(self.viz_grid)
        self.density = np.exp(log_density)

        peak_indices, _ = find_peaks(
            self.density, prominence=np.max(self.density) * 0.01 if self.density.size > 0 else 0.01
        )

        self.peaks = self.viz_grid.flatten()[peak_indices]
        self.peak_saliences = self.density[peak_indices]

    def get_plot_data(self) -> dict:
        """Returns all necessary data for plotting."""
        return {
            "grid": self.viz_grid.flatten(),
            "density": self.density,
            "peaks": self.peaks,
            "peak_saliences": self.peak_saliences,
        }


# --- Streamlit Application ---

st.set_page_config(layout="wide")
st.sidebar.title("Hierarchical KDE Boundary Explorer")


# --- Data Loading ---
@st.cache_resource
def get_dataset(manifest_path="~/data/salami/metadata.csv"):
    """Loads the dataset object, returns None if not found."""
    manifest_path = os.path.expanduser(manifest_path)
    if not os.path.exists(manifest_path):
        st.sidebar.error(f"Manifest not found: {manifest_path}")
        return None
    try:
        return bnl.data.Dataset(manifest_path=manifest_path)
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def load_boundary_contour(_dataset, track_id):
    """Loads a specific track's boundary contour."""
    if _dataset is None or track_id is None:
        return None
    try:
        track = _dataset[track_id]
        est = track.load_annotation("adobe-mu1gamma9")
        return ops.boundary_salience(est, strategy="depth")
    except Exception as e:
        st.warning(f"Failed to load data for track {track_id}: {e}")
        return None


slm_ds = get_dataset()

# --- Sidebar Controls ---
track_id = st.sidebar.selectbox("Select Track ID", slm_ds.track_ids, index=8)
boundary_contour = load_boundary_contour(slm_ds, track_id)
initial_times = np.array([b.time for b in boundary_contour.boundaries])
initial_saliences = np.array([b.salience for b in boundary_contour.boundaries])

st.sidebar.write("### Stage 1: Time Grouping")
# Configuration for Time Slider (Linear Scale)
time_bw_min, time_bw_max, time_bw_default, time_bw_step = 0.05, 7.0, 1.0, 0.05
time_bw_options = np.round(np.arange(0, time_bw_max, time_bw_step) + time_bw_min, 2)
# Find the closest default value in the new options to ensure it's a valid choice
time_bw_val = min(time_bw_options, key=lambda x: abs(x - time_bw_default))

time_bandwidth = st.sidebar.select_slider(
    "Time KDE Bandwidth (σ)",
    options=time_bw_options,
    value=time_bw_val,
    format_func=lambda x: f"{x:.2f} s",  # Format as seconds with 2 decimal places
    help="Controls the smoothness of the temporal density estimate.",
)


st.sidebar.write("### Stage 2: Salience Quantization")
# Configuration for Salience Slider (Log Scale)
sal_bw_min_log, sal_bw_max_log = np.log10(0.00001), np.log10(0.01)
sal_bw_options = np.logspace(sal_bw_min_log, sal_bw_max_log, num=50)
default_sal_bw = 0.0005
sal_bw_val = min(sal_bw_options, key=lambda x: abs(x - default_sal_bw))

salience_bandwidth = st.sidebar.select_slider(
    "Salience KDE Bandwidth (Log Scale)",
    options=sal_bw_options,
    value=sal_bw_val,
    format_func=lambda x: f"{x * 1e6:.0f}μ",  # Format for readability
    help="Controls the smoothness of the salience density estimate. The slider is on a log scale for finer control.",
)


# --- App Description ---
st.sidebar.markdown("---")
st.sidebar.info(
    """
This app performs a two-stage analysis.
The main plot merges temporal boundaries using a KDE.
The narrow plot on the right acts as a marginal histogram,
taking the *saliences* of the merged peaks and performing a second KDE to find significant salience levels.
"""
)


# --- Core Computations ---
@st.cache_resource
def get_time_explorer(times, saliences):
    return KDEBoundaryExplorer(times, saliences)


time_explorer = get_time_explorer(initial_times, initial_saliences)

# --- Core Logic ---
# Stage 1: Analyze time points
time_explorer.update_bandwidth(time_bandwidth)
time_plot_data = time_explorer.get_plot_data()
resulting_saliences = time_plot_data["peak_saliences"]

# Stage 2: Analyze the saliences from stage 1
salience_plot_data = {
    "grid": np.array([]),
    "density": np.array([]),
    "peaks": np.array([]),
    "peak_saliences": np.array([]),
}
quantized_saliences = np.array([])

if resulting_saliences.size > 0:
    salience_explorer = KDEBoundaryExplorer(resulting_saliences)
    salience_explorer.update_bandwidth(salience_bandwidth)
    salience_plot_data = salience_explorer.get_plot_data()
    salience_levels = salience_plot_data["peaks"]

    if salience_levels.size > 0:
        closest_level_indices = np.argmin(np.abs(salience_levels[:, np.newaxis] - resulting_saliences), axis=0)
        quantized_saliences = salience_levels[closest_level_indices]


# --- Plotting Functions ---
def make_interactive_plot():
    """Creates the main interactive plot with two subplots."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.02)

    # --- Traces for Main Plot (Time) ---
    fig.add_trace(go.Scatter(name="Time KDE Density", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(
        go.Scatter(name="Original Peaks", mode="markers", marker=dict(color="grey", size=8, symbol="diamond-open")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(name="Quantized Peaks", mode="markers", marker=dict(color="red", size=6, symbol="circle")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            name="Time Data Rug",
            mode="markers",
            marker_symbol="line-ns",
            marker=dict(color="rgba(170, 170, 170, 0.8)", size=12, line_width=1.5),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Traces for Marginal Plot (Salience) ---
    fig.add_trace(go.Scatter(name="Salience KDE Density", line=dict(color="mediumseagreen")), row=1, col=2)
    fig.add_trace(
        go.Scatter(name="Salience Levels", mode="markers", marker=dict(color="red", size=6, symbol="circle")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            name="Salience Data Rug",
            mode="markers",
            marker_symbol="line-ew",
            marker=dict(color="rgba(170, 170, 170, 0.8)", size=12, line_width=1.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # --- Layout ---
    fig.update_layout(
        title_text="Stage 1: Temporal Merging (Left) & Stage 2: Salience Quantization (Right)",
        height=450,  # Reduced height
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
        xaxis2=dict(zeroline=False),
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Density / Salience", row=1, col=1)
    fig.update_xaxes(title_text="Density", row=1, col=2)
    return fig


@st.cache_data
def compute_persistence_data(p_initial_times, p_initial_saliences, time_bw_params, sal_bw_range):
    """
    Pre-computes the number of peaks across the entire parameter space.
    This function is cached; it will re-run only when the track's data changes.
    """
    time_bw_range = np.arange(time_bw_params[0], time_bw_params[1], time_bw_params[2])
    # sal_bw_range is now passed in directly, no need to compute it here.

    # Use a local explorer instance for this computation, based on the specific track's data
    _time_explorer = KDEBoundaryExplorer(p_initial_times, p_initial_saliences)

    time_persistence = []
    salience_persistence = np.zeros((len(sal_bw_range), len(time_bw_range)))

    for i, t_bw in enumerate(time_bw_range):
        _time_explorer.update_bandwidth(t_bw)
        time_peaks = _time_explorer.get_plot_data()["peaks"]
        time_persistence.append(len(time_peaks))

        saliences = _time_explorer.get_plot_data()["peak_saliences"]
        if saliences.size > 0:
            # OPTIMIZATION: Initialize explorer once per time-bandwidth setting
            salience_explorer = KDEBoundaryExplorer(saliences)
            for j, s_bw in enumerate(sal_bw_range):
                salience_explorer.update_bandwidth(s_bw)
                salience_peaks = salience_explorer.get_plot_data()["peaks"]
                salience_persistence[j, i] = len(salience_peaks)

    return time_bw_range, sal_bw_range, time_persistence, salience_persistence


def make_persistence_plot(
    time_bw_range, sal_bw_range, time_persistence, salience_persistence, current_time_bw, current_sal_bw
):
    """Creates the persistence landscape and heatmap, arranged vertically to share the x-axis."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.3, 0.7],
        subplot_titles=(
            "Number of Temporal Boundaries",
            "Number of Salience Levels",
        ),
    )

    # Plot 1 (Top): Time Persistence (Number of Boundaries vs. Time Bandwidth)
    fig.add_trace(
        go.Scatter(x=time_bw_range, y=time_persistence, mode="lines", name="Time Peaks", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_vline(x=current_time_bw, line_width=2, line_dash="dash", line_color="red", name="Time BW", row=1, col=1)
    fig.update_yaxes(title_text="Num Boundaries", row=1, col=1)

    # Plot 2 (Bottom): Salience Persistence Heatmap
    fig.add_trace(
        go.Heatmap(
            z=salience_persistence,
            x=time_bw_range,
            y=sal_bw_range,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Num Levels"),
        ),
        row=2,
        col=1,
    )
    # Add cursors to the heatmap
    fig.add_vline(x=current_time_bw, line_width=2, line_dash="dash", line_color="red", name="Time BW", row=2, col=1)
    fig.add_hline(y=current_sal_bw, line_width=2, line_dash="dash", line_color="red", name="Salience BW", row=2, col=1)

    # Update axes and layout
    fig.update_xaxes(title_text="Time Bandwidth (σ)", row=2, col=1)
    fig.update_yaxes(title_text="Salience Bandwidth", row=2, col=1, type="log")
    fig.update_layout(
        title_text="Peak (left) and Level (right) vs. Bandwidth Persistence Landscape",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --- Main Display ---

# --- Interactive Analysis Plot ---
if "fig" not in st.session_state:
    st.session_state.fig = make_interactive_plot()

with st.session_state.fig.batch_update():
    # --- Main plot data (Time) ---
    st.session_state.fig.data[0].x, st.session_state.fig.data[0].y = time_plot_data["grid"], time_plot_data["density"]
    st.session_state.fig.data[1].x, st.session_state.fig.data[1].y = (
        time_plot_data["peaks"],
        time_plot_data["peak_saliences"],
    )
    st.session_state.fig.data[2].x, st.session_state.fig.data[2].y = time_plot_data["peaks"], quantized_saliences
    st.session_state.fig.data[3].x, st.session_state.fig.data[3].y = initial_times, np.zeros_like(initial_times)

    # --- Marginal plot data (Salience) ---
    st.session_state.fig.data[4].x, st.session_state.fig.data[4].y = (
        salience_plot_data["density"],
        salience_plot_data["grid"],
    )
    st.session_state.fig.data[5].x, st.session_state.fig.data[5].y = (
        salience_plot_data["peak_saliences"],
        salience_plot_data["peaks"],
    )
    st.session_state.fig.data[6].x, st.session_state.fig.data[6].y = (
        np.zeros_like(resulting_saliences),
        resulting_saliences,
    )


st.plotly_chart(st.session_state.fig, use_container_width=True)


# --- Persistence Landscape Plot ---
# Efficiently create and update the persistence plot.
# The expensive data computation is cached. The plot itself is stored in session
# state, and only the crosshair positions are updated on slider interaction.

# Compute or retrieve from cache
time_bw_params = (time_bw_min, time_bw_max, time_bw_step)
with st.spinner("Computing persistence landscape... This may take a moment."):
    time_bw_range, sal_bw_range, time_persistence, salience_persistence = compute_persistence_data(
        initial_times, initial_saliences, time_bw_params, sal_bw_options
    )

# Create the plot if it's the first run or if the track has changed
if "fig_persist" not in st.session_state or st.session_state.get("persistence_track_id") != track_id:
    fig_persist = make_persistence_plot(
        time_bw_range, sal_bw_range, time_persistence, salience_persistence, time_bandwidth, salience_bandwidth
    )
    st.session_state.fig_persist = fig_persist
    st.session_state.persistence_track_id = track_id
else:
    fig_persist = st.session_state.fig_persist

# On every interaction, retrieve the figure and just update the crosshairs
with fig_persist.batch_update():
    # These shapes correspond to the vline and hline calls in make_persistence_plot,
    # assuming the order is [vline_top, vline_bottom, hline_bottom].
    fig_persist.layout.shapes[0].x0 = time_bandwidth
    fig_persist.layout.shapes[0].x1 = time_bandwidth
    fig_persist.layout.shapes[1].x0 = time_bandwidth
    fig_persist.layout.shapes[1].x1 = time_bandwidth
    fig_persist.layout.shapes[2].y0 = salience_bandwidth
    fig_persist.layout.shapes[2].y1 = salience_bandwidth

st.plotly_chart(fig_persist, use_container_width=True)
