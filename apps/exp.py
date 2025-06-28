"""
SALAMI Explorer - Streamlit App using BNL

An interactive web application for exploring the SALAMI music dataset with audio playback,
visualization, and metadata browsing; powered by BNL.

Features:
- Browse 1,400+ music tracks with rich metadata (title, artist, duration)
- Stream MP3 audio directly from cloud storage (Cloudflare R2)
- Real-time waveform and MFCC visualization
- Dual data source support: Cloud (default) or local filesystem
- Cached data loading for optimal performance

Usage:
    pixi run exp

Data Sources:
- Cloud: Streams data directly from R2 bucket using online manifest
- Local: Supports SALAMI-style datasets with a path-based manifest containing relative asset paths

Architecture:
- Cloud-native: Streams data from R2 bucket without local storage
- Efficient caching: Uses Streamlit's @st.cache_data and @st.cache_resource
- Robust error handling: Graceful fallbacks for missing assets
- Real-time analysis: On-demand audio processing with librosa
"""

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

import bnl

# --- Page and Data Configuration ---
st.set_page_config(layout="wide")
st.title("SALAMI Explorer")

# --- Data Source and Path Configuration ---
R2_BUCKET_PUBLIC_URL = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"
CLOUD_MANIFEST_URL = f"{R2_BUCKET_PUBLIC_URL}/manifest_cloud_boolean.csv"
LOCAL_MANIFEST_PATH = os.path.expanduser("~/data/salami/metadata.csv")

# Let user choose data source
data_source_option = st.sidebar.radio(
    "Select Data Source:",
    ("Cloud (R2)", "Local Filesystem"),
)

# Manual session state management for data source
if "data_source_choice" not in st.session_state:
    st.session_state.data_source_choice = data_source_option
elif data_source_option != st.session_state.data_source_choice:
    st.session_state.data_source_choice = data_source_option
    # Clear everything when data source changes
    preserve_keys = ["data_source_choice"]  # Add other critical keys here if needed
    for key in list(st.session_state.keys()):
        if key not in preserve_keys:
            del st.session_state[key]
    st.cache_data.clear()
    st.rerun()


# --- Data Loading Functions (Cached for performance) ---
@st.cache_resource
def get_dataset(source_type: str):
    """Loads and caches the dataset object based on selected source."""
    if source_type == "Cloud (R2)":
        try:
            # Load from the online cloud manifest - Dataset will auto-detect it's a cloud source
            return bnl.data.Dataset(CLOUD_MANIFEST_URL)
        except Exception as e:
            st.error(f"Failed to load cloud manifest from: {CLOUD_MANIFEST_URL}")
            st.error(f"Error: {e}")
            st.error("Please check your internet connection and verify the cloud service is available.")
            st.stop()

    elif source_type == "Local Filesystem":
        try:
            # The Dataset class now infers the source type from the local path
            return bnl.data.Dataset(LOCAL_MANIFEST_PATH)
        except Exception as e:
            st.error(f"Failed to load LOCAL manifest from: {LOCAL_MANIFEST_PATH} \n Error: {e}")
            st.stop()
    else:
        st.error("Invalid data source selected.")
        st.stop()


# Get the current dataset based on user's choice
current_dataset = get_dataset(st.session_state.data_source_choice)


@st.cache_data
def load_track_data(_dataset, track_id):
    """Loads track data and populates track.info with metadata like title/artist."""
    track = _dataset[track_id]

    # Load audio using the Track's built-in method
    waveform, sr = track.load_audio()

    return track, waveform, sr


@st.cache_data
def create_audio_analysis_plot(waveform, sr):
    """Creates a matplotlib figure with waveform and MFCC plots."""
    if waveform is None or sr is None:
        return None

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


# --- UI Layout & App State ---
def reset_track_state():
    """Resets flags when a new track is selected, triggering a data reload."""
    st.session_state.track_loaded = False
    st.session_state.is_stabilized = False


# Sidebar for parameter tuning.
# Define this before the logic that uses its state.
with st.sidebar:
    # This selectbox initializes `st.session_state.track_id` on the first run.
    if not current_dataset.track_ids:
        st.warning("No track IDs found in the selected dataset.")
        st.stop()

    st.selectbox(
        "Select a SALAMI track id:",
        current_dataset.track_ids,
        key="track_id",
        on_change=reset_track_state,
    )
    # Initialize track_id to first available if not set or invalid
    if st.session_state.get("track_id") not in current_dataset.track_ids:
        st.session_state.track_id = current_dataset.track_ids[0]
        reset_track_state()

    # Show current status
    st.write(f"📊 **Data source:** {st.session_state.data_source_choice}")
    st.write(f"🎯 **Track:** {st.session_state.track_id}")

    if st.session_state.get("track_loaded", False) and hasattr(st.session_state, "track"):
        tinfo = st.session_state.track.info
        st.write(f"Title: {tinfo.get('title', 'N/A')}")  # Use .get for safety
        st.write(f"Artist: {tinfo.get('artist', 'N/A')}")

    st.header("Tuning Parameters (Placeholder)")
    bw_peak = st.slider("Bandwidth for Peak Picking", 0.1, 5.0, 1.0, 0.1)
    bw_group = st.slider("Bandwidth for Depth Grouping", 0.1, 5.0, 1.0, 0.1)


# --- Core Logic ---
# Initialize state for first run if they don't exist.
if "track_loaded" not in st.session_state:
    st.session_state.track_loaded = False
if "is_stabilized" not in st.session_state:  # May not be needed anymore
    st.session_state.is_stabilized = False


# Load track data when needed
if not st.session_state.track_loaded:
    with st.spinner(f"Loading track {st.session_state.track_id}..."):
        track, waveform, sr = load_track_data(current_dataset, st.session_state.track_id)
        analysis_plot = create_audio_analysis_plot(waveform, sr) if waveform is not None else None

    st.session_state.track = track
    st.session_state.analysis_plot = analysis_plot
    st.session_state.track_loaded = True

if st.session_state.track_loaded and not st.session_state.is_stabilized:
    # Data has just been loaded. Force one immediate rerun.
    # This "stabilizes" the UI, preventing widgets like st.audio from
    # resetting when a slider is moved for the first time.
    st.session_state.is_stabilized = True
    st.rerun()


# Main content area.
if st.session_state.get("track_loaded") and hasattr(st.session_state, "track"):
    track = st.session_state.track
    st.header(f"Track: {track.info.get('title', st.session_state.track_id)}")

    # --- Audio player: find the audio path from track.info ---
    audio_url_or_path = None
    for key, value in track.info.items():
        if key.startswith("audio_") and key.endswith("_path"):
            audio_url_or_path = value
            break

    if audio_url_or_path:
        st.audio(str(audio_url_or_path))  # Ensure it's a string for st.audio

        if st.session_state.analysis_plot:
            st.pyplot(st.session_state.analysis_plot)
        else:
            st.warning("Could not generate audio analysis plot (no waveform data).")
    else:
        st.warning(f"No audio asset found for track {st.session_state.track_id} in the manifest.")
        st.json(track.manifest_row.to_dict())  # Display what assets the manifest says it has

    # --- Annotation visualization ---
    if track.has_annotations:
        st.header("Annotations")
        st.write(f"Available annotation keys: {list(track.annotations.keys())}")
        # Attempt to load and plot the 'reference' annotation by default if it exists
        if "reference" in track.annotations:
            try:
                # Using the new load_annotation method
                ref_annotation = track.load_annotation("reference")
                st.write(f"Loaded 'reference' annotation (type: {type(ref_annotation)}).")
                if hasattr(ref_annotation, 'plot'):
                    # Assuming Hierarchy and Segmentation objects have a .plot() method
                    # that returns a matplotlib Figure.
                    fig = ref_annotation.plot()
                    st.pyplot(fig)
                else:
                    st.write("Loaded annotation object does not have a direct .plot() method.")
            except Exception as e:
                st.error(f"Error loading or plotting 'reference' annotation: {e}")
        else:
            st.info("No 'reference' annotation key found for this track.")
            st.write("Consider adding UI to select other annotation types and IDs if needed.")

    else:
        st.warning("No annotations found for this track.")

else:
    st.warning("Please select a track or wait for data to load.")
