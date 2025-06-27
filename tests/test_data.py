"""Tests for the manifest-based data loading core (path-based)."""

import io
from pathlib import Path

import pandas as pd
import pytest

from bnl import data

# --- Constants for Mocking ---
MOCK_CLOUD_URL_BASE = "https://mock-r2-bucket.com"
MOCK_LOCAL_DATA_ROOT = "test_data_root"


# --- Helper Functions ---
def create_mock_manifest_df(
    track_ids: list[str], assets_info: dict[str, dict[str, str]]
) -> pd.DataFrame:
    """Creates a mock DataFrame for a path-based manifest."""
    records = []
    all_cols = {"track_id"}
    for assets in assets_info.values():
        all_cols.update(assets.keys())

    for tid in track_ids:
        record = {"track_id": str(tid), **assets_info.get(str(tid), {})}
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure all columns are present, filling missing with None (pd.NA)
    for col in all_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure consistent column order
    sorted_cols = ["track_id"] + sorted(
        [col for col in df.columns if col != "track_id"]
    )
    return df[sorted_cols]


# --- Fixtures ---
@pytest.fixture
def mock_local_manifest_file(tmp_path: Path) -> Path:
    """Creates a mock local path-based manifest and corresponding dummy files."""
    manifest_dir = tmp_path / MOCK_LOCAL_DATA_ROOT
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "metadata.csv"

    # Create dummy asset files
    (manifest_dir / "audio" / "1").mkdir(parents=True, exist_ok=True)
    (manifest_dir / "audio" / "1" / "audio.mp3").touch()
    jams_dir = manifest_dir / "jams"
    jams_dir.mkdir(parents=True, exist_ok=True)
    with open(jams_dir / "1.jams", "w") as f:
        f.write(
            '{"file_metadata": {"title": "MockTitle", "artist": "MockArtist", "duration": 10.0}}'
        )

    assets_info = {
        "1": {
            "audio_mp3_path": "audio/1/audio.mp3",
            "annotation_reference_path": "jams/1.jams",
        },
        "2": {"audio_mp3_path": "audio/2/audio.wav"},
    }
    df = create_mock_manifest_df(track_ids=["1", "2", "3"], assets_info=assets_info)
    df.to_csv(manifest_file, index=False)
    return manifest_file


@pytest.fixture
def mock_cloud_manifest_file(requests_mock) -> str:
    """Mocks a cloud-based manifest file served via HTTP."""
    manifest_url = f"{MOCK_CLOUD_URL_BASE}/manifest.csv"
    assets_info = {
        "101": {
            "audio_mp3_path": f"{MOCK_CLOUD_URL_BASE}/audio/101.mp3",
            "annotation_reference_path": f"{MOCK_CLOUD_URL_BASE}/jams/101.jams",
        },
        "102": {"audio_mp3_path": f"{MOCK_CLOUD_URL_BASE}/audio/102.mp3"},
    }
    df = create_mock_manifest_df(track_ids=["101", "102"], assets_info=assets_info)
    requests_mock.get(manifest_url, text=df.to_csv(index=False))
    return manifest_url


# --- Test Core Functionality ---
def test_dataset_init_local(mock_local_manifest_file: Path):
    """Test initializing a Dataset from a local manifest."""
    dataset = data.Dataset(mock_local_manifest_file)
    assert dataset.data_source_type == "local"
    assert dataset.dataset_root == mock_local_manifest_file.parent
    assert dataset.list_tids() == ["1", "2", "3"]


def test_dataset_init_cloud(mock_cloud_manifest_file: str):
    """Test initializing a Dataset from a cloud manifest URL."""
    dataset = data.Dataset(mock_cloud_manifest_file)
    assert dataset.data_source_type == "cloud"
    assert hasattr(dataset, "dataset_root") is False  # No local root for cloud
    assert dataset.list_tids() == ["101", "102"]


def test_dataset_init_file_not_found():
    with pytest.raises(FileNotFoundError):
        data.Dataset("non/existent/manifest.csv")


def test_load_track_local(mock_local_manifest_file: Path):
    """Test loading a track and accessing its info from a local dataset."""
    dataset = data.Dataset(mock_local_manifest_file)
    track = dataset.load_track("1")

    assert track.track_id == "1"
    assert "Track(track_id='1'" in repr(track)
    assert "num_assets=2" in repr(track)

    # Check that info resolves relative paths and parses JAMS metadata
    info = track.info
    expected_audio_path = mock_local_manifest_file.parent / "audio/1/audio.mp3"
    assert info["audio_mp3_path"] == expected_audio_path
    assert info["title"] == "MockTitle"

    # Mock librosa.load to test the audio loading call
    class MockLibrosa:
        def load(self, path, sr, mono):
            assert path == expected_audio_path
            return (pd.Series([1, 2, 3]), 22050)

    original_librosa_load = data.librosa.load
    data.librosa.load = MockLibrosa().load
    y, sr = track.load_audio()
    assert y is not None
    assert sr == 22050
    data.librosa.load = original_librosa_load


def test_load_track_cloud(mock_cloud_manifest_file: str, requests_mock):
    """Test loading a track, parsing JAMS, and loading audio from a cloud dataset."""
    dataset = data.Dataset(mock_cloud_manifest_file)
    track = dataset.load_track("101")

    # Check basic info and URL construction
    assert track.track_id == "101"
    expected_audio_url = f"{MOCK_CLOUD_URL_BASE}/audio/101.mp3"
    assert track.info["audio_mp3_path"] == expected_audio_url

    # Mock JAMS request and check metadata parsing
    jams_url = track.manifest_row["annotation_reference_path"]
    requests_mock.get(
        jams_url,
        text='{"file_metadata": {"title": "CloudTitle", "artist": "CloudArtist", "duration": 20.0}}',
    )
    assert track.info["title"] == "CloudTitle"

    # Mock audio request and check audio loading
    mock_audio_content = io.BytesIO(b"fake_audio_data")
    requests_mock.get(expected_audio_url, body=mock_audio_content)

    class MockLibrosa:
        def load(self, buffer, sr, mono):
            assert isinstance(buffer, io.BytesIO)
            return (pd.Series([1, 2, 3]), 44100)

    original_librosa_load = data.librosa.load
    data.librosa.load = MockLibrosa().load
    y, sr = track.load_audio()
    assert y is not None
    assert sr == 44100
    data.librosa.load = original_librosa_load


def test_load_track_with_missing_assets(mock_local_manifest_file: Path):
    """Ensure that missing asset paths are handled gracefully."""
    dataset = data.Dataset(mock_local_manifest_file)
    track = dataset.load_track("3")  # Track 3 has no assets in the mock manifest
    assert "num_assets=0" in repr(track)
    assert pd.isna(track.manifest_row["audio_mp3_path"])
    y, sr = track.load_audio()
    assert y is None
    assert sr is None


def test_load_nonexistent_track(mock_local_manifest_file: Path):
    """Test that loading a track_id not in the manifest raises a ValueError."""
    dataset = data.Dataset(mock_local_manifest_file)
    with pytest.raises(ValueError, match="Track ID 'nonexistent' not found"):
        dataset.load_track("nonexistent")


def test_dataset_handles_non_numeric_track_ids(tmp_path: Path):
    """Ensure Dataset correctly sorts non-numeric track IDs lexically."""
    manifest_file = tmp_path / "metadata.csv"
    # Contains a non-numeric track_id ('a_track'), which will cause the
    # numeric sort to fail and fall back to a lexical sort.
    df = pd.DataFrame({"track_id": ["10", "2", "a_track"]})
    df.to_csv(manifest_file, index=False)

    dataset = data.Dataset(manifest_file)
    # Expect a lexical sort ('10', '2', 'a_track'), not a numeric one.
    assert dataset.list_tids() == ["10", "2", "a_track"]
