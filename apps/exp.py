import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import io
import requests
from pathlib import Path

import bnl

# --- Page and Data Configuration ---
st.set_page_config(layout="wide")
st.title("BNL Data Explorer")

# --- Data Source Configuration ---
R2_BUCKET_PUBLIC_URL = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"
CLOUD_MANIFEST_URL = f"{R2_BUCKET_PUBLIC_URL}/manifest_cloud_boolean.csv"
LOCAL_DATASET_ROOT = (
    "~/data/salami"  # Example local path, user might need to change this
)
LOCAL_MANIFEST_PATH = f"{LOCAL_DATASET_ROOT}/metadata.csv"

# Let user choose data source
data_source_option = st.sidebar.radio(
    "Select Data Source:",
    ("Cloud (R2)", "Local Filesystem"),
    key="data_source_choice",
    on_change=lambda: st.session_state.clear(),  # Clear cache on source change
)


# --- Data Loading Functions (Cached for performance) ---
@st.cache_resource
def get_dataset(source_type: str):
    """Loads and caches the dataset object based on selected source."""
    if source_type == "Cloud (R2)":
        manifest_path = CLOUD_MANIFEST_URL
        try:
            # For cloud, cloud_base_url is where assets are, not necessarily where manifest is.
            return bnl.data.Dataset(
                manifest_path,
                data_source_type="cloud",
                cloud_base_url=R2_BUCKET_PUBLIC_URL,
            )
        except Exception as e:
            st.error(f"Failed to load CLOUD manifest from: {manifest_path}")
            st.error(f"Error: {e}")
            st.stop()
    elif source_type == "Local Filesystem":
        manifest_path = LOCAL_MANIFEST_PATH
        try:
            return bnl.data.Dataset(manifest_path, data_source_type="local")
        except FileNotFoundError:
            st.error(f"Local manifest not found at: {manifest_path}")
            st.error(
                "Ensure your local SALAMI dataset is structured correctly and manifest exists."
            )
            st.info(
                f"Expected structure: \n{LOCAL_DATASET_ROOT}/\n"
                "  ├── audio/\n  ├── jams/\n  └── metadata.csv"
            )
            st.stop()
        except Exception as e:
            st.error(f"Failed to load LOCAL manifest from: {manifest_path}")
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.error("Invalid data source selected.")
        st.stop()


# Get the current dataset based on user's choice
with st.spinner(f"Loading dataset from {st.session_state.data_source_choice}..."):
    current_dataset = get_dataset(st.session_state.data_source_choice)


@st.cache_data  # Cache based on the dataset object's identity (implicitly via current_dataset)
def get_tids(_dataset):  # Pass dataset to make caching aware of it
    """Fetches the list of all available track IDs."""
    return _dataset.list_tids()


@st.cache_data
def load_track_data(_dataset, track_id):  # Pass dataset
    """
    Loads track data, handling cloud/local paths and populating necessary info.
    """
    track = _dataset.load_track(track_id)

    # First, load annotations to populate track.info with metadata like title/artist
    # This is necessary because the cloud manifest may not contain this info directly.
    try:
        track.load_annotations()
    except Exception as e:
        st.warning(f"Could not load annotations for track {track_id}: {e}")
        # Even if annotations fail, try to proceed. Populate info from manifest.
        if not track.info:
            track.info = track.manifest_row.to_dict()

    # Find the audio URL from track.info (works for both boolean and direct path formats)
    audio_url_or_path = None

    # Try track.info first (boolean format with reconstructed paths)
    for key, value in track.info.items():
        if key.startswith("audio_") and key.endswith("_path"):
            audio_url_or_path = value
            break

    # Fallback to manifest_row (direct path format)
    if not audio_url_or_path:
        for key, value in track.manifest_row.items():
            if key.startswith("audio_") and key.endswith("_path"):
                audio_url_or_path = value
                break

    waveform, sr = (None, None)
    if audio_url_or_path:
        try:
            # Load directly from URL using requests and BytesIO - Cloudflare-compatible
            response = requests.get(str(audio_url_or_path))
            response.raise_for_status()
            waveform, sr = librosa.load(
                io.BytesIO(response.content), sr=None, mono=True
            )
        except Exception as e:
            st.warning(f"Librosa failed to load audio from: {audio_url_or_path}")
            st.error(f"Error: {e}")

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
    # Pass current_dataset to get_tids
    available_tids = get_tids(current_dataset)
    if not available_tids:
        st.warning("No track IDs found in the selected dataset.")
        st.stop()

    st.selectbox(
        "Select a SALAMI track id:",
        available_tids,
        key="track_id",  # This will be reset if available_tids changes due to source switch
        on_change=reset_track_state,
    )

    # Ensure track_id is valid for the current dataset, select first if not
    if (
        "track_id" not in st.session_state
        or st.session_state.track_id not in available_tids
    ):
        st.session_state.track_id = available_tids[0]
        reset_track_state()  # Ensure data reloads for the new default track

    if st.session_state.get("track_loaded", False) and hasattr(
        st.session_state, "track"
    ):
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


# Primary logic flow for loading and stabilizing data.
if not st.session_state.track_loaded:
    # A new track has been selected, or it's the first run, or data source changed.
    spinner_message = f"Loading track data ({st.session_state.data_source_choice})..."
    with st.spinner(spinner_message):
        # Pass current_dataset to load_track_data
        track, waveform, sr = load_track_data(
            current_dataset, st.session_state.track_id
        )
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
if st.session_state.get("track_loaded") and hasattr(st.session_state, "track"):
    track = st.session_state.track
    st.header(f"Track: {track.info.get('title', st.session_state.track_id)}")

    # Audio player: find the audio path from track.info or manifest_row
    audio_url_or_path = None

    # Try track.info first (boolean format with reconstructed paths)
    for key, value in track.info.items():
        if key.startswith("audio_") and key.endswith("_path"):
            audio_url_or_path = value
            break

    # Fallback to manifest_row (direct path format)
    if not audio_url_or_path:
        for key in track.manifest_row.keys():
            if key.startswith("audio_") and key.endswith("_path"):
                audio_url_or_path = track.manifest_row[key]
                break

    if audio_url_or_path:
        st.audio(str(audio_url_or_path))  # Ensure it's a string for st.audio

        if st.session_state.analysis_plot:
            st.pyplot(st.session_state.analysis_plot)
        else:
            st.warning("Could not generate audio analysis plot (no waveform data).")
    else:
        st.warning(
            f"No audio asset found for track {st.session_state.track_id} in the manifest."
        )
        st.json(
            track.manifest_row.to_dict()
        )  # Display what assets the manifest says it has

else:
    st.warning("Please select a track or wait for data to load.")
