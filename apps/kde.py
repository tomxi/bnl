import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List


# --- Data Structures ---
@dataclass
class RatedBoundary:
    """A simple data class for our weighted time points."""

    time: float
    salience: float


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
st.title("Hierarchical KDE Boundary Explorer")
st.markdown("""
This app performs a two-stage analysis. The main plot merges temporal boundaries using a KDE. The narrow plot on the right acts as a marginal histogram, taking the *saliences* of the merged peaks and performing a second KDE to find significant salience levels.
""")

# --- Sample Data Generation ---
coarse_boundaries = [RatedBoundary(2.0, 1 / 3), RatedBoundary(10.0, 1 / 3), RatedBoundary(15.0, 1 / 3)]
fine_boundaries = [RatedBoundary(t, 1 / 300) for t in np.linspace(start=0, stop=20, num=300)]
all_boundaries = coarse_boundaries + fine_boundaries
initial_times = np.array([b.time for b in all_boundaries])
initial_saliences = np.array([b.salience for b in all_boundaries])

# --- Sidebar Controls ---
st.sidebar.header("Stage 1: Time Grouping")
time_bandwidth = st.sidebar.slider(
    "Time KDE Bandwidth (σ)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.05,
    help="Controls the smoothness of the temporal density estimate.",
)

st.sidebar.header("Stage 2: Salience Quantization")
salience_bandwidth = st.sidebar.slider(
    "Salience KDE Bandwidth",
    min_value=0.001,
    max_value=0.05,
    value=0.005,
    step=0.001,
    format="%.3f",
    help="Controls the smoothness of the salience density estimate.",
)


# --- App Instantiation ---
@st.cache_resource
def get_time_explorer(times, saliences):
    return KDEBoundaryExplorer(times, saliences)


time_explorer = get_time_explorer(initial_times, initial_saliences)

# --- Core Logic ---
# Stage 1: Analyze time points
time_explorer.update_bandwidth(time_bandwidth)
time_plot_data = time_explorer.get_plot_data()
resulting_saliences = time_plot_data["peak_saliences"]

# FIX: Check if there are peaks from stage 1 before running stage 2
if resulting_saliences.size > 0:
    # Stage 2: Analyze the saliences from stage 1
    salience_explorer = KDEBoundaryExplorer(resulting_saliences)
    salience_explorer.update_bandwidth(salience_bandwidth)
    salience_plot_data = salience_explorer.get_plot_data()
else:
    # If no peaks, create empty data for the second plot
    salience_plot_data = {
        "grid": np.array([]),
        "density": np.array([]),
        "peaks": np.array([]),
        "peak_saliences": np.array([]),
    }


# --- EFFICIENT PLOTTING WITH SUBPLOTS ---
# 1. Initialize the subplot figure ONCE and store it in session state.
if "fig" not in st.session_state:
    st.session_state.fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.02
    )
    # Add empty traces that we will update later
    # Main plot (time)
    st.session_state.fig.add_trace(go.Scatter(name="Time KDE Density", line=dict(color="royalblue")), row=1, col=1)
    st.session_state.fig.add_trace(
        go.Scatter(name="Time Peaks", mode="markers", marker=dict(color="red", size=8, symbol="x")), row=1, col=1
    )

    # Marginal plot (salience)
    st.session_state.fig.add_trace(
        go.Scatter(name="Salience KDE Density", line=dict(color="mediumseagreen")), row=1, col=2
    )
    st.session_state.fig.add_trace(
        go.Scatter(name="Salience Levels", mode="markers", marker=dict(color="orange", size=8, symbol="cross")),
        row=1,
        col=2,
    )

    # Set the layout properties that don't change
    st.session_state.fig.update_layout(
        title_text="Stage 1: Temporal Merging (Left) & Stage 2: Salience Quantization (Right)",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    )
    st.session_state.fig.update_xaxes(title_text="Time", row=1, col=1)
    st.session_state.fig.update_yaxes(title_text="Density / Salience", row=1, col=1)
    st.session_state.fig.update_xaxes(title_text="Density", row=1, col=2)

# 2. On every script run, update the DATA of the existing figure.
with st.session_state.fig.batch_update():
    # Update main plot data
    st.session_state.fig.data[0].x = time_plot_data["grid"]
    st.session_state.fig.data[0].y = time_plot_data["density"]
    st.session_state.fig.data[1].x = time_plot_data["peaks"]
    st.session_state.fig.data[1].y = time_plot_data["peak_saliences"]

    # Update marginal plot data (swapping x and y for vertical orientation)
    st.session_state.fig.data[2].x = salience_plot_data["density"]
    st.session_state.fig.data[2].y = salience_plot_data["grid"]
    st.session_state.fig.data[3].x = salience_plot_data["peak_saliences"]
    st.session_state.fig.data[3].y = salience_plot_data["peaks"]

# 3. Display the single figure from session state.
st.plotly_chart(st.session_state.fig, use_container_width=True)

# --- Raw Data Expanders ---
expander_col1, expander_col2 = st.columns([0.8, 0.2])
with expander_col1:
    with st.expander("Show Time Peak Data"):
        st.write({"bandwidth": time_bandwidth, "peak_times": np.round(time_plot_data["peaks"], 3).tolist()})
with expander_col2:
    with st.expander("Show Salience Level Data"):
        st.write(
            {"bandwidth": salience_bandwidth, "salience_levels": np.round(salience_plot_data["peaks"], 4).tolist()}
        )
