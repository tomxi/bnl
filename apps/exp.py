import random

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
R2_BUCKET_BASE = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"
SALAMI_MANIFEST_PATH = f"{R2_BUCKET_BASE}/manifest_cloud.csv"


# --- Data Loading Functions (Cached for performance) ---
@st.cache_resource
def get_dataset():
    """Loads and caches the dataset object."""
    try:
        return bnl.data.Dataset(SALAMI_MANIFEST_PATH)
    except Exception as e:
        st.error(f"Failed to load cloud manifest from: {SALAMI_MANIFEST_PATH}")
        st.error(f"Error: {e}")
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
    if waveform is None or sr is None:
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 4), sharex=True, constrained_layout=True
    )
    librosa.display.waveshow(waveform, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel(None)

    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
    img = librosa.display.specshow(mfccs, x_axis="time", y_axis="mel", sr=sr, ax=ax2)
    ax2.set_title("MFCC")

    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    return fig


# --- UI Layout & App State ---
def reset_track_state():
    """Resets flags when a new track is selected, triggering a data reload."""
    st.session_state.track_loaded = False
    st.session_state.is_stabilized = False


# Sidebar for parameter tuning.
# Define this before the logic that uses its state.
with st.sidebar:
    # This selectbox initializes `st.session_state.track_id` on the first run.
    st.selectbox(
        "Select a SALAMI track id:",
        get_tids(),
        key="track_id",
        on_change=reset_track_state,
    )
    if st.session_state.track_loaded:
        tinfo = st.session_state.track.info
        st.write(f"{tinfo['title']}")
        st.write(f"*by* {tinfo['artist']}")
    st.header("Tuning Parameters")
    bw_peak = st.slider("Bandwidth for Peak Picking", 0.1, 5.0, 1.0, 0.1)
    bw_group = st.slider("Bandwidth for Depth Grouping", 0.1, 5.0, 1.0, 0.1)


# --- Core Logic ---
# Initialize state for first run.
# This ensures the keys exist before they are accessed.
if "track_loaded" not in st.session_state:
    st.session_state.track_loaded = False
if "is_stabilized" not in st.session_state:
    st.session_state.is_stabilized = False

# Primary logic flow for loading and stabilizing data.
if not st.session_state.track_loaded:
    # A new track has been selected, or it's the first run.
    with st.spinner("Loading track data from cloud..."):
        track, waveform, sr = load_track_data(st.session_state.track_id)
        analysis_plot = (
            create_audio_analysis_plot(waveform, sr) if waveform is not None else None
        )

    st.session_state.track = track
    st.session_state.waveform = waveform
    st.session_state.sr = sr
    st.session_state.analysis_plot = analysis_plot
    st.session_state.track_loaded = True

if st.session_state.track_loaded and not st.session_state.is_stabilized:
    # Data has just been loaded. Force one immediate rerun.
    # This "stabilizes" the UI, preventing widgets like st.audio from
    # resetting when a slider is moved for the first time.
    st.session_state.is_stabilized = True
    st.rerun()


# Main content area.
if st.session_state.get("track_loaded"):
    # Display track metadata and the pre-computed analysis plot.
    track = st.session_state.track
    st.header(f"Track {st.session_state.track.info['title']}")

    # Audio player
    if audio_url := track.info.get("audio_path"):
        st.audio(audio_url)

        # Display audio analysis if available
        if st.session_state.analysis_plot:
            st.pyplot(st.session_state.analysis_plot)
        else:
            st.warning("Could not generate audio analysis plot.")
    else:
        st.warning("No audio file found for this track.")


else:
    # This part should ideally not be reached if initialization works correctly,
    # but serves as a fallback.
    st.warning("Please select a track to begin.")
