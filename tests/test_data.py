"""Tests for the data loading module."""

import pytest
from pathlib import Path

from bnl import data, core



# Use a real track from the test fixtures
TRACK_ID = "8"
FIXTURES_DIR = Path("tests/fixtures")
TEST_MANIFEST = FIXTURES_DIR / "test-manifest.csv"


@pytest.fixture(scope="module")
def dataset() -> data.Dataset:
    """Provides a Dataset instance for testing."""
    return data.Dataset(TEST_MANIFEST)


@pytest.fixture(scope="module")
def dataset_cloud() -> data.Dataset:
    """Provides a Dataset instance for testing."""
    bucket_url = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"
    return data.Dataset(manifest_path=f"{bucket_url}/manifest_cloud_boolean.csv")


def test_dataset_loading(dataset: data.Dataset):
    """Tests basic dataset and track loading."""
    assert dataset.data_location == "local"
    assert len(dataset) > 0
    assert isinstance(dataset.track_ids, list)
    assert all(isinstance(tid, str) for tid in dataset.track_ids)
    track = dataset[TRACK_ID]
    assert track.track_id == TRACK_ID
    assert len(track.info) > 0


def test_dataset_manifest_not_found():
    """Test that a FileNotFoundError is raised for a non-existent manifest."""
    with pytest.raises(FileNotFoundError):
        data.Dataset(manifest_path="/non/existent/path/manifest.csv")


def test_track_properties_and_errors(dataset: data.Dataset):
    """Test track properties and error handling for loading annotations."""
    track = dataset[TRACK_ID]

    # Test __repr__
    assert f"Track(track_id='{TRACK_ID}'" in repr(track)
    assert "num_assets=" in repr(track)

    # Test for annotation type that doesn't exist
    with pytest.raises(ValueError, match="not available"):
        track.load_annotation("non_existent_type")

    # Test for file type that is not supported
    track.info["annotation_fake_type_path"] = "/fake/path/file.txt"
    with pytest.raises(NotImplementedError, match="Unsupported file type"):
        track.load_annotation("fake_type")


def test_load_cloud_est(dataset: data.Dataset, dataset_cloud: data.Dataset):
    test_track_est = dataset[TRACK_ID].load_annotation("adobe-mu1gamma1")
    cloud_track_est = dataset_cloud[TRACK_ID].load_annotation("adobe-mu1gamma1")

    assert isinstance(test_track_est, core.MultiSegment)
    assert len(test_track_est.layers) >= 1
    assert test_track_est == cloud_track_est


def test_load_cloud_ref(dataset: data.Dataset, dataset_cloud: data.Dataset):
    test_track_ref = dataset[TRACK_ID].load_annotation("reference")
    cloud_track_ref = dataset_cloud[TRACK_ID].load_annotation("reference")

    assert isinstance(test_track_ref, core.MultiSegment)
    assert len(test_track_ref.layers) == 2
    assert test_track_ref == cloud_track_ref
