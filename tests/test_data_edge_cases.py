"""Edge case tests for data.py."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from bnl import data


def test_parse_jams_metadata_error():
    """Tests JAMS parsing with a nonexistent path."""
    assert data._parse_jams_metadata("/nonexistent/path.jams") == {}


def test_dataset_missing_track_id_column(tmp_path):
    """Tests `Dataset` with a missing `track_id` column."""
    manifest_file = tmp_path / "bad.csv"
    pd.DataFrame({"other": ["1"]}).to_csv(manifest_file, index=False)

    with pytest.raises(ValueError, match="track_id"):
        data.Dataset(manifest_file)


def test_track_load_audio_no_assets(tmp_path):
    """Tests audio loading when no audio assets are available."""
    manifest_file = tmp_path / "metadata.csv"
    pd.DataFrame({"track_id": ["1"]}).to_csv(manifest_file, index=False)

    dataset = data.Dataset(manifest_file)
    y, sr = dataset["1"].load_audio()
    assert y is None and sr is None


def test_track_hierarchy_missing_annotation(tmp_path):
    """Tests loading a missing annotation."""
    manifest_file = tmp_path / "metadata.csv"
    pd.DataFrame({"track_id": ["1"]}).to_csv(manifest_file, index=False)

    dataset = data.Dataset(manifest_file)
    with pytest.raises(ValueError, match="not available"):
        dataset["1"].load_annotation("missing")


def test_dataset_iteration(tmp_path):
    """Tests `Dataset` iteration."""
    manifest_file = tmp_path / "metadata.csv"
    pd.DataFrame({"track_id": ["1", "2"]}).to_csv(manifest_file, index=False)

    dataset = data.Dataset(manifest_file)
    assert len(dataset) == 2
    assert [t.track_id for t in dataset] == ["1", "2"]


def test_track_repr_and_info(tmp_path):
    """Tests `Track` string representation and `info` property."""
    manifest_file = tmp_path / "metadata.csv"
    pd.DataFrame({"track_id": ["1"], "artist": ["Test"]}).to_csv(manifest_file, index=False)

    dataset = data.Dataset(manifest_file)
    track = dataset["1"]
    assert "Track" in repr(track)
    assert track.info["track_id"] == "1"


def test_reconstruct_path_errors():
    """Tests `_reconstruct_path` error cases."""
    manifest_file = Path("fake.csv")
    with patch("bnl.data.pd.read_csv", return_value=pd.DataFrame({"track_id": ["1"]})):
        dataset = data.Dataset(manifest_file)

    with pytest.raises(ValueError, match="Unknown"):
        dataset._reconstruct_path("1", "bad_type", "bad_subtype")


def test_dataset_cloud_request_error():
    """Tests `Dataset` with a cloud request error."""
    with patch("requests.get", side_effect=requests.RequestException("Network error")):
        with pytest.raises(requests.RequestException):
            data.Dataset("https://example.com/manifest.csv")
