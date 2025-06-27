import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

import bnl

# --- Page and Data Configuration ---
# st.set_page_config(layout="wide")
st.title("BNL Data Explorer")

# --- Data Source Configuration ---
R2_BUCKET_PUBLIC_URL = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"
CLOUD_MANIFEST_URL = f"{R2_BUCKET_PUBLIC_URL}/manifest_cloud.csv"
LOCAL_DATASET_ROOT = "./SALAMI_selected"  # Example local path, user might need to change this
LOCAL_MANIFEST_PATH = f"{LOCAL_DATASET_ROOT}/metadata.csv"

# Let user choose data source
data_source_option = st.sidebar.radio(
    "Select Data Source:",
    ("Cloud (R2)", "Local Filesystem"),
    key="data_source_choice",
    on_change=lambda: st.session_state.clear(),  # Clear cache on source change
)


# --- Data Loading Functions (Cached for performance) ---
@st.cache_resource  # Use _experimental_singleton or cache_resource for dataset object
def get_dataset(source_type: str):
    """Loads and caches the dataset object based on selected source."""
    if source_type == "Cloud (R2)":
        manifest_path = CLOUD_MANIFEST_URL
        try:
            st.write(f"Loading CLOUD dataset from: {manifest_path}")
            # For cloud, cloud_base_url is where assets are, not necessarily where manifest is.
            return bnl.data.Dataset(
                manifest_path, data_source_type="cloud", cloud_base_url=R2_BUCKET_PUBLIC_URL
            )
        except Exception as e:
            st.error(f"Failed to load CLOUD manifest from: {manifest_path}")
            st.error(f"Error: {e}")
            st.stop()
    elif source_type == "Local Filesystem":
        manifest_path = LOCAL_MANIFEST_PATH
        try:
            st.write(f"Loading LOCAL dataset from: {manifest_path}")
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
current_dataset = get_dataset(st.session_state.data_source_choice)


@st.cache_data  # Cache based on the dataset object's identity (implicitly via current_dataset)
def get_tids(_dataset):  # Pass dataset to make caching aware of it
    """Fetches the list of all available track IDs."""
    return _dataset.list_tids()


@st.cache_data
def load_track_data(_dataset, track_id):  # Pass dataset
    """Loads the track object, audio waveform, and sample rate for a given ID."""
    track = _dataset.load_track(track_id)
    # Accessing track.info will trigger path reconstruction if not already done
    st.write(f"Track info for {track_id}: {track.info}")  # Debug: show reconstructed paths
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
    if "track_id" not in st.session_state or st.session_state.track_id not in available_tids:
        st.session_state.track_id = available_tids[0]
        reset_track_state()  # Ensure data reloads for the new default track

    if st.session_state.get("track_loaded", False) and hasattr(st.session_state, "track"):
        tinfo = st.session_state.track.info
        st.write(f"Title: {tinfo.get('title', 'N/A')}")  # Use .get for safety
        st.write(f"Artist: {tinfo.get('artist', 'N/A')}")
    st.header("Tuning Parameters (Placeholder)")
    # bw_peak = st.slider("Bandwidth for Peak Picking", 0.1, 5.0, 1.0, 0.1)
    # bw_group = st.slider("Bandwidth for Depth Grouping", 0.1, 5.0, 1.0, 0.1)


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
        track, waveform, sr = load_track_data(current_dataset, st.session_state.track_id)
        analysis_plot = (
            create_audio_analysis_plot(waveform, sr) if waveform is not None else None
        )

    st.session_state.track = track
    st.session_state.waveform = waveform
    st.session_state.sr = sr
    st.session_state.analysis_plot = analysis_plot
    st.session_state.track_loaded = True
    # st.session_state.is_stabilized = False # Reset stabilization flag
    st.rerun()  # Rerun to update UI after loading

# This stabilization logic might need adjustment or removal depending on Streamlit
# version and behavior.
# if st.session_state.track_loaded and not st.session_state.is_stabilized:
#     st.session_state.is_stabilized = True
#     st.rerun()


# Main content area.
if st.session_state.get("track_loaded") and hasattr(st.session_state, "track"):
    track = st.session_state.track
    st.header(f"Track: {track.info.get('title', st.session_state.track_id)}")

    # Audio player: find the audio path from track.info
    # It could be 'audio_mp3_path', 'audio_wav_path', etc.
    audio_url_or_path = None
    for key in track.info:
        if key.startswith("audio_") and key.endswith("_path"):
            audio_url_or_path = track.info[key]
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
        st.json(track.manifest_row.to_dict())  # Display what assets the manifest says it has

else:
    st.warning("Please select a track or wait for data to load.")
