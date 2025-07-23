"""Tests for the data loading module."""

from pathlib import Path

import pytest

from bnl.core import MultiSegment
from bnl.data import Dataset

# Use a real track from the test fixtures
TRACK_ID = "8"
FIXTURES_DIR = Path("tests/fixtures")
TEST_MANIFEST = FIXTURES_DIR / "test-manifest.csv"


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Provides a Dataset instance for testing."""
    return Dataset(TEST_MANIFEST)


@pytest.fixture(scope="module")
def dataset_cloud() -> Dataset:
    """Provides a Dataset instance for testing."""
    bucket_url = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"
    return Dataset(manifest_path=f"{bucket_url}/manifest_cloud_boolean.csv")


def test_dataset_loading(dataset: Dataset):
    """Tests basic dataset and track loading."""
    assert len(dataset) > 0
    track = dataset[TRACK_ID]
    assert track.track_id == TRACK_ID
    assert len(track.info) > 0


def test_load_cloud_est(dataset: Dataset, dataset_cloud: Dataset):
    test_track_est = dataset[TRACK_ID].load_annotation("adobe-mu1gamma1")
    cloud_track_est = dataset_cloud[TRACK_ID].load_annotation("adobe-mu1gamma1")

    assert isinstance(test_track_est, MultiSegment)
    assert len(test_track_est.layers) >= 1
    assert test_track_est == cloud_track_est


def test_load_cloud_ref(dataset: Dataset, dataset_cloud: Dataset):
    test_track_ref = dataset[TRACK_ID].load_annotation("reference")
    cloud_track_ref = dataset_cloud[TRACK_ID].load_annotation("reference")

    assert isinstance(test_track_ref, MultiSegment)
    assert len(test_track_ref.layers) == 2
    assert test_track_ref == cloud_track_ref
