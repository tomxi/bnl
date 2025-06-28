"""Tests for the manifest-based data loading core (path-based)."""

import io
from pathlib import Path

import pandas as pd
import pytest
import requests

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
    jams_content_local = """{
        "file_metadata": {"title": "MockTitle", "artist": "MockArtist", "duration": 10.0},
        "annotations": [{
            "namespace": "multi_segment",
            "data": [{"time": 0, "duration": 10.0, "value": {"label": "segment_A", "level": 0}, "confidence": 1.0}],
            "sandbox": {},
            "annotation_metadata": {"corpus": "local"}
        }]
    }"""
    with open(jams_dir / "1.jams", "w") as f:
        f.write(jams_content_local)

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
    with open(mock_cloud_manifest_file) as f:
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
    with open(mock_cloud_manifest_file) as f:
        manifest_content = f.read()
    requests_mock.get(cloud_manifest_url, text=manifest_content)

    dataset = data.Dataset(cloud_manifest_url)
    track = dataset["101"]

    # --- Test JAMS and Audio Loading ---
    # 1. Construct expected URLs first
    expected_jams_url = f"{MOCK_CLOUD_URL_BASE}/ref-jams/101.jams"
    expected_audio_url = f"{MOCK_CLOUD_URL_BASE}/slm-dataset/101/audio.mp3"

    # 2. Set up mocks BEFORE accessing track.info, which triggers the requests
    jams_content_cloud = """{
        "file_metadata": {"title": "CloudTitle", "artist": "CloudArtist", "duration": 20.0},
        "annotations": [{
            "namespace": "multi_segment",
            "data": [{"time": 0, "duration": 20.0, "value": {"label": "segment_B", "level": 0}, "confidence": 1.0}],
            "sandbox": {},
            "annotation_metadata": {"corpus": "cloud"}
        }]
    }"""
    requests_mock.get(expected_jams_url, text=jams_content_cloud)
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


def test_load_hierarchy_no_multisegment_annotation(mock_local_manifest_file: Path, mocker):
    """Test loading hierarchy when JAMS file has no multi_segment annotation."""
    dataset = data.Dataset(mock_local_manifest_file)
    track = dataset["1"]  # Track "1" has a JAMS file

    # Mock jams.load to return a JAMS object with other annotations but no multi_segment
    mock_jams_obj = mocker.MagicMock(spec=data.jams.JAMS)
    mock_anno_other_namespace = mocker.MagicMock(spec=data.jams.Annotation)
    mock_anno_other_namespace.namespace = "pitch_contour"
    mock_jams_obj.annotations = [mock_anno_other_namespace]

    mocker.patch("jams.load", return_value=mock_jams_obj)

    with pytest.raises(ValueError, match="No multi_segment annotation found"):
        track.load_hierarchy("reference")


def test_parse_jams_metadata_error_handling(mocker, capsys):
    """Test error handling in _parse_jams_metadata."""
    # Test with a non-existent file - should not print a warning, just return empty
    data._parse_jams_metadata("non_existent.jams")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    # Test with a URL that returns an error
    # Ensure this mock is specific or reset if other tests use requests.get
    mocker.patch("requests.get", side_effect=requests.exceptions.RequestException("Mocked HTTP error"))
    data._parse_jams_metadata("http://example.com/error.jams")
    captured = capsys.readouterr()
    assert "Warning: Could not parse JAMS metadata" in captured.out  # Changed to .out
    assert "Mocked HTTP error" in captured.out  # Changed to .out

    # Test with a JAMS file that causes a parsing error
    mock_response = mocker.MagicMock()
    mock_response.text = "invalid jams content"
    # Reset requests.get mock if it was set with side_effect earlier in the same test scope
    mocker.patch("requests.get", return_value=mock_response)
    mocker.patch("jams.load", side_effect=Exception("Mocked JAMS load error"))
    data._parse_jams_metadata("http://example.com/bad.jams")
    captured = capsys.readouterr()
    assert "Warning: Could not parse JAMS metadata" in captured.out  # Changed to .out
    assert "Mocked JAMS load error" in captured.out  # Changed to .out


def test_track_properties_and_methods(mock_local_manifest_file: Path):
    """Test various Track properties and methods like has_annotations and annotations."""
    dataset = data.Dataset(mock_local_manifest_file)
    track1 = dataset["1"]  # Has annotation_reference
    track2 = dataset["2"]  # No annotations

    assert track1.has_annotations
    assert "reference" in track1.annotations
    assert isinstance(track1.annotations["reference"], Path)

    assert not track2.has_annotations
    assert track2.annotations == {}

    # Test repr for a track with no 'has_*' columns (e.g. if manifest was empty beyond track_id)
    empty_manifest_row = pd.Series({"track_id": "empty"})
    empty_track = data.Track(track_id="empty", manifest_row=empty_manifest_row, dataset=dataset)
    assert "num_assets=0" in repr(empty_track)


def test_load_hierarchy_local(mock_local_manifest_file: Path, mocker):
    """Test loading a hierarchy from a local JAMS file."""
    dataset = data.Dataset(mock_local_manifest_file)
    track = dataset["1"]

    # Mock bnl.data.Hierarchy.from_jams, let actual jams.load run
    mock_hierarchy_from_jams = mocker.patch("bnl.data.Hierarchy.from_jams")

    hierarchy = track.load_hierarchy("reference")

    # The real jams.load is called by track.load_hierarchy inside the method.
    # Then Hierarchy.from_jams is called with the multi_segment annotation from the loaded JAMS.
    mock_hierarchy_from_jams.assert_called_once()
    loaded_jams_annotation_arg = mock_hierarchy_from_jams.call_args[0][0]
    assert isinstance(loaded_jams_annotation_arg, data.jams.Annotation)
    assert loaded_jams_annotation_arg.namespace == "multi_segment"
    assert hierarchy == mock_hierarchy_from_jams.return_value

    # Test error if annotation type is not available
    with pytest.raises(ValueError, match="Annotation type 'nonexistent' not available"):
        track.load_hierarchy("nonexistent")


def test_load_hierarchy_cloud(mock_cloud_manifest_file: Path, requests_mock, mocker):
    """Test loading a hierarchy from a cloud JAMS file."""
    cloud_manifest_url = f"{MOCK_CLOUD_URL_BASE}/manifest_cloud.csv"
    with open(mock_cloud_manifest_file) as f:
        manifest_content = f.read()
    requests_mock.get(cloud_manifest_url, text=manifest_content)

    dataset = data.Dataset(cloud_manifest_url)
    track = dataset["101"]  # Has annotation_reference

    expected_jams_url = f"{MOCK_CLOUD_URL_BASE}/ref-jams/101.jams"
    # Ensure this JAMS content is the same as used in test_load_track_cloud or compatible
    jams_content_cloud_hierarchy = """{
        "file_metadata": {"title": "CloudTitle", "artist": "CloudArtist", "duration": 20.0},
        "annotations": [{
            "namespace": "multi_segment",
            "data": [{"time": 0, "duration": 20.0, "value": {"label": "segment_B", "level": 0}, "confidence": 1.0}],
            "sandbox": {},
            "annotation_metadata": {"corpus": "cloud_hier"}
        }]
    }"""
    requests_mock.get(expected_jams_url, text=jams_content_cloud_hierarchy)  # Mock JAMS download

    # Mock bnl.data.Hierarchy.from_jams
    # We want the actual jams.load to be called with the content from requests_mock
    mock_hierarchy_from_jams = mocker.patch("bnl.data.Hierarchy.from_jams")

    # Temporarily mock jams.load to inspect its argument if needed, but allow real call
    real_jams_load = data.jams.load

    def sideload_jams_load_and_capture(string_io_arg, **kwargs):
        # This allows us to check that jams.load was indeed called with stringIO
        assert isinstance(string_io_arg, io.StringIO)
        # Then call the real jams.load
        return real_jams_load(string_io_arg, **kwargs)

    mocker.patch("jams.load", side_effect=sideload_jams_load_and_capture)

    hierarchy = track.load_hierarchy("reference")

    # Assert that our mocked Hierarchy.from_jams was called
    # The argument to it would be the multi_segment JAMS Annotation object
    mock_hierarchy_from_jams.assert_called_once()
    jams_annotation_arg = mock_hierarchy_from_jams.call_args[0][0]
    assert isinstance(jams_annotation_arg, data.jams.Annotation)
    assert jams_annotation_arg.namespace == "multi_segment"
    assert hierarchy == mock_hierarchy_from_jams.return_value

    # Test error if JAMS download fails
    requests_mock.get(expected_jams_url, status_code=404)
    with pytest.raises(requests.exceptions.HTTPError):
        track.load_hierarchy("reference")


def test_dataset_iteration(mock_local_manifest_file: Path):
    """Test iterating over a dataset."""
    dataset = data.Dataset(mock_local_manifest_file)
    tracks = [t for t in dataset]
    assert len(tracks) == len(dataset.track_ids)
    assert all(isinstance(t, data.Track) for t in tracks)
    assert [t.track_id for t in tracks] == dataset.track_ids


def test_reconstruct_path_errors(mock_local_manifest_file: Path):
    """Test error conditions for _reconstruct_path."""
    dataset = data.Dataset(mock_local_manifest_file)
    with pytest.raises(ValueError, match="Unknown local asset structure"):
        dataset._reconstruct_path("1", "unknown_type", "subtype")

    # Switch to cloud temporarily to test cloud error
    dataset.data_location = "cloud"
    dataset.base_url = "http://foo.bar"
    with pytest.raises(ValueError, match="Unknown cloud asset structure"):
        dataset._reconstruct_path("1", "unknown_type", "subtype")
    with pytest.raises(ValueError, match="Unknown cloud asset structure"):
        dataset._reconstruct_path("1", "audio", "unknown_subtype")


def test_dataset_init_value_errors(tmp_path: Path, mocker):
    """Test ValueErrors during Dataset initialization."""
    # Test manifest without track_id column
    no_track_id_manifest = tmp_path / "no_track_id.csv"
    pd.DataFrame({"some_col": [1, 2, 3]}).to_csv(no_track_id_manifest, index=False)
    with pytest.raises(ValueError, match="Manifest must contain a 'track_id' column"):
        data.Dataset(no_track_id_manifest)

    # Test manifest file not found (local)
    with pytest.raises(FileNotFoundError, match="Manifest file not found at: non_existent_local_manifest.csv"):
        data.Dataset("non_existent_local_manifest.csv")

    # Test manifest file not found (cloud) - this will be caught by requests.get in Dataset init
    # This is already implicitly tested by test_dataset_init_file_not_found if it uses a cloud URL
    # but we can make it explicit for pd.read_csv part.
    # For the pd.read_csv part of cloud loading, an HTTPError from requests.get would prevent it.
    # So this specific FileNotFoundError from pd.read_csv is harder to isolate for cloud.

    # The "Unsupported data_location" else clause in Dataset.__init__ is currently unreachable
    # because data_location is always determined as 'cloud' or 'local' based on the scheme.
    # A scheme that is not http/https will result in 'local'.
    # Thus, the ValueError for unsupported data_location in __init__ cannot be tested
    # without significantly altering the logic or making the scheme parsing more complex.


def test_track_audio_load_failure_cases(mock_local_manifest_file: Path, monkeypatch, capsys, mocker):
    """Test failure cases for Track.load_audio()."""
    dataset = data.Dataset(mock_local_manifest_file)
    track_no_audio_in_manifest = dataset["3"]  # Track 3 has no audio asset in manifest
    y, sr = track_no_audio_in_manifest.load_audio()
    assert y is None
    assert sr is None

    track_with_audio_file = dataset["1"]
    # Simulate librosa load error
    mocker.patch.object(data.librosa, "load", side_effect=Exception("Librosa mock error"))
    y, sr = track_with_audio_file.load_audio()
    assert y is None
    assert sr is None
    captured = capsys.readouterr()
    assert "Failed to load audio" in captured.out
    assert "Librosa mock error" in captured.out

    # Simulate HTTP error for cloud loading
    dataset.data_location = "cloud"
    dataset.base_url = MOCK_CLOUD_URL_BASE

    cloud_track_manifest_row = dataset.manifest.iloc[0].copy()
    cloud_track_manifest_row["has_audio_mp3"] = True
    cloud_track_manifest_row["has_annotation_reference"] = True

    track_cloud_audio = data.Track("101", cloud_track_manifest_row, dataset)

    original_requests_get = requests.get

    expected_jams_url_for_info = f"{MOCK_CLOUD_URL_BASE}/ref-jams/{track_cloud_audio.track_id}.jams"

    def mock_get_for_jams_info(url, **kwargs):
        if url == expected_jams_url_for_info:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'{"file_metadata": {"title":"Cloud JAMS", "artist":"Cloud Artist", '
                b'"duration":123.0}, "annotations":[]}'
            )
            return resp
        return original_requests_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", mock_get_for_jams_info)
    _ = track_cloud_audio.info

    assert track_cloud_audio.info["title"] == "Cloud JAMS"

    def mock_get_for_audio_error(url, **kwargs):
        if "audio_mp3_path" in track_cloud_audio.info and url == track_cloud_audio.info["audio_mp3_path"]:
            response = requests.Response()
            response.status_code = 404
            response.reason = "Audio Not Found"
            response.url = url
            raise requests.exceptions.HTTPError(f"Simulated 404 HTTP Error for {url}", response=response)
        pytest.fail(f"Unexpected requests.get call to {url} during audio load failure simulation.")
        return None

    monkeypatch.setattr(requests, "get", mock_get_for_audio_error)

    y_cloud, sr_cloud = track_cloud_audio.load_audio()
    assert y_cloud is None
    assert sr_cloud is None

    captured_cloud_audio_error = capsys.readouterr()
    assert "Failed to load audio" in captured_cloud_audio_error.out
    assert "404" in captured_cloud_audio_error.out
    assert track_cloud_audio.info["audio_mp3_path"] in captured_cloud_audio_error.out

    monkeypatch.setattr(requests, "get", original_requests_get)
    dataset.data_location = "local"

    track_with_audio_file.info.pop("audio_mp3_path", None)
    y_no_path, sr_no_path = track_with_audio_file.load_audio()
    assert y_no_path is None
    assert sr_no_path is None


def test_adobe_path_reconstruction_local_and_cloud(tmp_path: Path):
    """Test specific Adobe path reconstruction logic."""
    manifest_file = tmp_path / "metadata.csv"
    df = pd.DataFrame(
        {
            "track_id": ["adobe_test"],
            "has_annotation_adobe-mu1gamma1": [True],
            "has_annotation_adobe-mu5gamma5": [True],
            "has_annotation_adobe-mu1gamma9": [True],
        }
    )
    df.to_csv(manifest_file, index=False)
    dataset = data.Dataset(manifest_file)

    # Local - test the actual asset types we support
    # For adobe-mu1gamma1 -> def_mu_0.1_gamma_0.1
    expected_local1 = (
        dataset.dataset_root / "adobe/def_mu_0.1_gamma_0.1" / "adobe_test.mp3.msdclasscsnmagic.json"
    ).resolve()
    assert str(dataset._reconstruct_path("adobe_test", "annotation", "adobe-mu1gamma1").resolve()) == str(
        expected_local1
    )

    # For adobe-mu5gamma5 -> def_mu_0.5_gamma_0.5
    expected_local2 = (
        dataset.dataset_root / "adobe/def_mu_0.5_gamma_0.5" / "adobe_test.mp3.msdclasscsnmagic.json"
    ).resolve()
    assert str(dataset._reconstruct_path("adobe_test", "annotation", "adobe-mu5gamma5").resolve()) == str(
        expected_local2
    )

    # For adobe-mu1gamma9 -> def_mu_0.1_gamma_0.9
    expected_local3 = (
        dataset.dataset_root / "adobe/def_mu_0.1_gamma_0.9" / "adobe_test.mp3.msdclasscsnmagic.json"
    ).resolve()
    assert str(dataset._reconstruct_path("adobe_test", "annotation", "adobe-mu1gamma9").resolve()) == str(
        expected_local3
    )

    # Cloud - test the actual asset types we support
    dataset.data_location = "cloud"
    dataset.base_url = "http://cloud.test"
    expected_cloud1 = "http://cloud.test/adobe21-est/def_mu_0.1_gamma_0.1/adobe_test.mp3.msdclasscsnmagic.json"
    expected_cloud2 = "http://cloud.test/adobe21-est/def_mu_0.5_gamma_0.5/adobe_test.mp3.msdclasscsnmagic.json"
    expected_cloud3 = "http://cloud.test/adobe21-est/def_mu_0.1_gamma_0.9/adobe_test.mp3.msdclasscsnmagic.json"
    assert dataset._reconstruct_path("adobe_test", "annotation", "adobe-mu1gamma1") == expected_cloud1
    assert dataset._reconstruct_path("adobe_test", "annotation", "adobe-mu5gamma5") == expected_cloud2
    assert dataset._reconstruct_path("adobe_test", "annotation", "adobe-mu1gamma9") == expected_cloud3
