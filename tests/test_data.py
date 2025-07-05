"""Tests for the data loading module."""

from pathlib import Path

import pytest

from bnl.core import MultiSegment
from bnl.data import Dataset

# Use a real track from the test fixtures
TRACK_ID = "8"
FIXTURES_DIR = Path("tests/fixtures")
TEST_MANIFEST = FIXTURES_DIR / "test_manifest.csv"


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Provides a Dataset instance for testing."""
    return Dataset(TEST_MANIFEST)


def test_dataset_loading(dataset: Dataset):
    """Tests basic dataset and track loading."""
    assert len(dataset) > 0
    track = dataset[TRACK_ID]
    assert track.track_id == TRACK_ID
    assert "annotation_ref_hier_jams_path" in track.info
    assert "annotation_adobe-mu1gamma1_path" in track.info


def test_load_reference_annotation(dataset: Dataset):
    """Tests loading a reference JAMS annotation."""
    track = dataset[TRACK_ID]
    annotation = track.load_annotation("ref_hier_jams")

    assert isinstance(annotation, MultiSegment)
    assert len(annotation.layers) == 2  # Based on the structure of 8.jams


def test_load_adobe_annotation(dataset: Dataset):
    """Tests loading an Adobe JSON annotation."""
    track = dataset[TRACK_ID]
    annotation = track.load_annotation("adobe-mu1gamma1")

    assert isinstance(annotation, MultiSegment)
    assert len(annotation.layers) > 0
