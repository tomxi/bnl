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
def create_mock_manifest_df(track_ids: list[str], assets_info: dict[str, dict[str, str]]) -> pd.DataFrame:
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
    sorted_cols = ["track_id"] + sorted([col for col in df.columns if col != "track_id"])
    return df[sorted_cols]


def create_mock_boolean_manifest_df(track_ids: list[str], assets_info: dict[str, list[str]]) -> pd.DataFrame:
    """Creates a mock DataFrame for a boolean manifest."""
    records = []
    all_cols = {"track_id"}
    # From the assets_info values (lists of column names), find all possible columns
    for track_assets in assets_info.values():
        all_cols.update(track_assets)

    for tid in track_ids:
        record = {"track_id": str(tid)}
        # For each possible asset column, mark it True if it's in this track's list
        for col in all_cols:
            if col != "track_id":
                record[col] = col in assets_info.get(str(tid), [])
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure consistent column order
    sorted_cols = ["track_id"] + sorted([col for col in df.columns if col != "track_id"])
    return df[sorted_cols]


# --- Fixtures ---
@pytest.fixture
def mock_local_manifest_file(tmp_path: Path) -> Path:
    """Creates a mock local BOOLEAN manifest and corresponding dummy files."""
    manifest_dir = tmp_path / MOCK_LOCAL_DATA_ROOT
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "metadata.csv"

    # Create dummy asset files that the loader will try to find
    (manifest_dir / "audio" / "1").mkdir(parents=True, exist_ok=True)
    (manifest_dir / "audio" / "1" / "audio.mp3").touch()
    jams_dir = manifest_dir / "jams"
    jams_dir.mkdir(parents=True, exist_ok=True)
    with open(jams_dir / "1.jams", "w") as f:
        f.write('{"file_metadata": {"title": "MockTitle", "artist": "MockArtist", "duration": 10.0}}')

    # Define which assets EXIST for each track_id via boolean flags
    assets_info = {
        "1": ["has_audio_mp3", "has_annotation_reference"],
        "2": ["has_audio_mp3"],  # Note: The actual file audio/2/audio.mp3 doesn't exist
        "3": [],  # Track with no assets
    }
    df = create_mock_boolean_manifest_df(track_ids=["1", "2", "3"], assets_info=assets_info)
    df.to_csv(manifest_file, index=False)
    return manifest_file


@pytest.fixture
def mock_cloud_manifest_file(tmp_path: Path) -> Path:
    """Creates a mock local BOOLEAN manifest for testing cloud logic."""
    manifest_file = tmp_path / "manifest_cloud.csv"
    assets_info = {
        "101": ["has_audio_mp3", "has_annotation_reference"],
        "102": ["has_audio_mp3"],
    }
    df = create_mock_boolean_manifest_df(track_ids=["101", "102"], assets_info=assets_info)
    df.to_csv(manifest_file, index=False)
    return manifest_file


# --- Test Core Functionality ---
def test_dataset_init_local(mock_local_manifest_file: Path):
    """Test initializing a Dataset from a local manifest."""
    dataset = data.Dataset(mock_local_manifest_file)
    assert dataset.data_location == "local"
    assert dataset.dataset_root == mock_local_manifest_file.parent
    assert dataset.track_ids == ["1", "2", "3"]


def test_dataset_init_cloud(mock_cloud_manifest_file: Path, requests_mock):
    """Test initializing a Dataset for cloud usage with a cloud manifest URL."""
    # Mock the cloud manifest URL
    cloud_manifest_url = f"{MOCK_CLOUD_URL_BASE}/manifest_cloud.csv"
    with open(mock_cloud_manifest_file, "r") as f:
        manifest_content = f.read()
    requests_mock.get(cloud_manifest_url, text=manifest_content)

    dataset = data.Dataset(cloud_manifest_url)
    assert dataset.data_location == "cloud"
    assert dataset.base_url == MOCK_CLOUD_URL_BASE
    assert dataset.track_ids == ["101", "102"]


def test_dataset_init_file_not_found():
    with pytest.raises(FileNotFoundError):
        data.Dataset("non/existent/manifest.csv")


def test_load_track_local(mock_local_manifest_file: Path, monkeypatch):
    """Test loading a track and accessing its info from a local dataset."""
    dataset = data.Dataset(mock_local_manifest_file)
    track = dataset["1"]

    assert track.track_id == "1"
    assert "Track(track_id='1'" in repr(track)
    assert "num_assets=2" in repr(track)

    # Check that info resolves relative paths and parses JAMS metadata
    info = track.info
    expected_audio_path = mock_local_manifest_file.parent / "audio/1/audio.mp3"
    assert info["audio_mp3_path"] == expected_audio_path
    assert info["annotation_reference_path"] == (mock_local_manifest_file.parent / "jams/1.jams")
    assert info["title"] == "MockTitle"

    # Mock librosa.load to test audio loading without a real audio file
    def mock_load(path, sr, mono):
        assert path == expected_audio_path
        return (pd.Series([1, 2, 3]), 22050)

    monkeypatch.setattr(data.librosa, "load", mock_load)
    y, sr = track.load_audio()
    assert y is not None
    assert sr is not None

    # Test audio loading for a track with a missing file (but present in manifest)
    track2 = dataset["2"]
    y2, sr2 = track2.load_audio()
    assert y2 is None
    assert sr2 is None


def test_load_track_cloud(mock_cloud_manifest_file: Path, requests_mock, monkeypatch):
    """Test loading a track, parsing JAMS, and loading audio from a cloud dataset."""
    # Mock the cloud manifest URL
    cloud_manifest_url = f"{MOCK_CLOUD_URL_BASE}/manifest_cloud.csv"
    with open(mock_cloud_manifest_file, "r") as f:
        manifest_content = f.read()
    requests_mock.get(cloud_manifest_url, text=manifest_content)

    dataset = data.Dataset(cloud_manifest_url)
    track = dataset["101"]

    # --- Test JAMS and Audio Loading ---
    # 1. Construct expected URLs first
    expected_jams_url = f"{MOCK_CLOUD_URL_BASE}/ref-jams/101.jams"
    expected_audio_url = f"{MOCK_CLOUD_URL_BASE}/slm-dataset/101/audio.mp3"

    # 2. Set up mocks BEFORE accessing track.info, which triggers the requests
    requests_mock.get(
        expected_jams_url,
        text='{"file_metadata": {"title": "CloudTitle", "artist": "CloudArtist", "duration": 20.0}}',
    )
    mock_audio_content = io.BytesIO(b"fake_audio_data")
    requests_mock.get(expected_audio_url, body=mock_audio_content)

    # 3. Now, access .info and .load_audio() to trigger the cached calls
    info = track.info
    assert info["annotation_reference_path"] == expected_jams_url
    assert info["audio_mp3_path"] == expected_audio_url
    assert info["title"] == "CloudTitle"

    # Mock librosa to check that it's called with the downloaded content
    def mock_load(buffer, sr, mono):
        assert isinstance(buffer, io.BytesIO)
        return (pd.Series([1, 2, 3]), 44100)

    monkeypatch.setattr(data.librosa, "load", mock_load)
    y, sr = track.load_audio()
    assert y is not None
    assert sr == 44100


def test_load_track_with_missing_assets(mock_local_manifest_file: Path):
    """Ensure that missing asset paths are handled gracefully."""
    dataset = data.Dataset(mock_local_manifest_file)
    track = dataset["3"]  # Track 3 has no assets in the manifest
    assert "num_assets=0" in repr(track)
    assert track.info == {"track_id": "3"}  # Info should be mostly empty
    y, sr = track.load_audio()
    assert y is None


def test_load_nonexistent_track(mock_local_manifest_file: Path):
    """Test that loading a track_id not in the manifest raises a ValueError."""
    dataset = data.Dataset(mock_local_manifest_file)
    with pytest.raises(ValueError, match="Track ID 'nonexistent' not found"):
        dataset["nonexistent"]


def test_dataset_handles_non_numeric_track_ids(tmp_path: Path):
    """Ensure Dataset correctly sorts non-numeric track IDs lexically."""
    manifest_file = tmp_path / "metadata.csv"
    # Contains a non-numeric track_id ('a_track'), which will cause the
    # numeric sort to fail and fall back to a lexical sort.
    df = pd.DataFrame({"track_id": ["10", "2", "a_track"], "has_audio_mp3": [True, True, False]})
    df.to_csv(manifest_file, index=False)

    dataset = data.Dataset(manifest_file)
    # Expect a lexical sort ('10', '2', 'a_track'), not a numeric one.
    assert dataset.track_ids == ["10", "2", "a_track"]
