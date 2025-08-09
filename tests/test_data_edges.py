"""Edge-case coverage for data module (bnl/src/bnl/data.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bnl import data

TEST_TID = "8"


@pytest.fixture(scope="module")
def ds_local() -> data.Dataset:
    return data.Dataset(Path(__file__).parent / "fixtures" / "test-manifest.csv")


@pytest.fixture(scope="module")
def ds_cloud() -> data.Dataset:
    # Uses the default online manifest
    return data.Dataset()


def test_track_refs_when_no_jam(ds_local: data.Dataset):
    """Cover Track.refs branch when there is no reference JAMS (jam is None)."""
    # Construct a Track with an empty Series so no `has_*` columns exist.
    dummy_row = pd.Series(dtype=object)
    tr = data.Track(track_id="dummy", manifest_row=dummy_row, dataset=ds_local)
    assert tr.jam is None  # covers data.py:L93
    assert tr.refs == {}  # covers data.py:L74


def test_load_jams_anno_no_matching_annotator(ds_local: data.Dataset):
    """Request a non-existent annotator to hit the ValueError path."""
    tr = ds_local[TEST_TID]
    with pytest.raises(ValueError, match="No annotator found"):
        tr.load_annotation("reference", annotator="__no_such_annotator__")  # covers L124


def test_fetch_content_file_not_found():
    with pytest.raises(FileNotFoundError):
        data.Track._fetch_content("/this/path/does/not/exist___bnl")  # covers L151


def test_track_ids_with_non_numeric_sorting(tmp_path: Path):
    """Create a small manifest mixing numeric and non-numeric track_ids to hit the
    ValueError branch and ensure lexicographic sorting is used (covers L204-205)."""
    manifest = tmp_path / "mixed-manifest.csv"
    manifest.write_text(
        """
track_id,has_annotation_reference
A,True
10,True
        """.strip()
    )
    ds = data.Dataset(manifest)
    # Should not raise; should fall back to lexicographic sort
    assert ds.track_ids == sorted(["A", "10"])  # covers L205


def test_format_adobe_params_else_branch():
    # Any subtype whose second dash-part isn't one of the three special cases
    assert data.Dataset._format_adobe_params("adobe-foobar") == "foobar"  # covers L232


def test_reconstruct_local_path_unknown_asset_raises(ds_local: data.Dataset):
    with pytest.raises(ValueError, match="Unknown local asset"):
        ds_local._reconstruct_local_path(TEST_TID, "unknown", "x")  # covers L248


def test_reconstruct_cloud_url_unknown_asset_raises(ds_cloud: data.Dataset):
    with pytest.raises(ValueError, match="Unknown cloud asset"):
        ds_cloud._reconstruct_cloud_url(TEST_TID, "unknown", "x")  # covers L264


def test_reconstruct_path_unknown_data_location(ds_local: data.Dataset):
    # Force an unknown data_location to trigger the guard
    object.__setattr__(ds_local, "data_location", "weird")
    with pytest.raises(ValueError, match="Unknown data location"):
        ds_local._reconstruct_path(TEST_TID, "audio", "mp3")  # covers L279
