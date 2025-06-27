import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

import bnl

# --- Page and Data Configuration ---
# st.set_page_config(layout="wide")
st.title("Monotonicity Casting Explorer")

# Define the path to the dataset manifest.
# This makes the data source explicit and easy to change.
SALAMI_MANIFEST_PATH = Path.home() / "data/salami/metadata.csv"


# --- Data Loading Functions (Cached for performance) ---
@st.cache_resource
def get_dataset():
    """Loads and caches the dataset object."""
    try:
        return bnl.data.Dataset(SALAMI_MANIFEST_PATH)
    except FileNotFoundError:
        st.error(f"SALAMI manifest not found at: {SALAMI_MANIFEST_PATH}")
        st.info("Please run `python scripts/build_manifest.py ~/data/salami` to generate it.")
        st.stop()


@st.cache_data
def get_tids():
    """Fetches the list of all available track IDs."""
    dataset = get_dataset()
    return dataset.list_tids()


@st.cache_data
def load_track_data(track_id):
    """Loads the track object, audio waveform, and sample rate for a given ID."""
    dataset = get_dataset()
    track = dataset.load_track(track_id)
    waveform, sr = track.load_audio()
    return track, waveform, sr


@st.cache_data
def create_audio_analysis_plot(waveform, sr):
    """Creates a matplotlib figure with waveform and MFCC plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, constrained_layout=True)
    librosa.display.waveshow(waveform, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel(None)

    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
    img = librosa.display.specshow(mfccs, x_axis="time", y_axis="mel", sr=sr, ax=ax2)
    ax2.set_title("MFCC")

    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    return fig


# --- App State & Core Logic ---
def reset_app_state():
    """Resets flags when a new track is selected, triggering a data reload."""
    st.session_state.track_loaded = False
    st.session_state.is_stabilized = False


# Initialize state on the first run.
if "track_id" not in st.session_state:
    st.session_state.track_id = random.choice(get_tids())
    reset_app_state()

# Primary logic flow for loading and stabilizing data.
if not st.session_state.track_loaded:
    # A new track has been selected; load its data.
    track, waveform, sr = load_track_data(st.session_state.track_id)
    analysis_plot = create_audio_analysis_plot(waveform, sr)

    st.session_state.track = track
    st.session_state.analysis_plot = analysis_plot
    st.session_state.track_loaded = True

if st.session_state.track_loaded and not st.session_state.is_stabilized:
    # Data has just been loaded. Force one immediate rerun.
    # This "stabilizes" the UI, preventing widgets like st.audio from
    # resetting when a slider is moved for the first time.
    st.session_state.is_stabilized = True
    st.rerun()


# --- UI Layout ---

# Sidebar for parameter tuning.
with st.sidebar:
    st.selectbox(
        "Select a SALAMI track id:",
        get_tids(),
        key="track_id",
        on_change=reset_app_state,
    )
    st.header("Tuning Parameters")
    bw_peak = st.slider("Bandwidth for Peak Picking", 0.1, 5.0, 1.0, 0.1)
    bw_group = st.slider("Bandwidth for Depth Grouping", 0.1, 5.0, 1.0, 0.1)

# Main content area.


if st.session_state.get("track_loaded"):
    # Display track metadata and the pre-computed analysis plot.
    track = st.session_state.track
    st.header(f"{st.session_state.track_id}: {track.info.get('title', 'N/A')}")

    if "audio_path" in track.info:
        st.audio(str(track.info["audio_path"]))
    else:
        st.warning("No audio file found for this track.")

    st.write(
        f"**Artist:** {track.info.get('artist', 'N/A')} | "
        f"**Duration:** {track.info.get('duration', 'N/A')}s"
    )
    st.pyplot(st.session_state.analysis_plot)

    # Placeholder for the downstream analysis UI.
    st.header("Downstream Analysis")
    st.info("This section updates when you move the sliders.")
    # Your core logic that uses bw_peak and bw_group goes here.

else:
    # This part should ideally not be reached if initialization works correctly,
    # but serves as a fallback.
    st.warning("Please select a track to begin.")
