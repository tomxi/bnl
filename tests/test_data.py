"""Tests for the manifest-based data loading core."""

from pathlib import Path

import pytest

from bnl import data

# Define the manifest path for tests. Assumes the manifest has been generated.
SALAMI_MANIFEST_PATH = Path.home() / "data/salami/metadata.csv"
SALAMI_DATA_AVAILABLE = SALAMI_MANIFEST_PATH.exists()


def test_track_repr() -> None:
    """Test `Track` representation doesn't crash."""
    track = data.Track(track_id="999", assets={}, dataset_root=Path("."))
    assert "999" in repr(track)


def test_dataset_init_file_not_found() -> None:
    """Test `Dataset` raises an error for a bad path."""
    with pytest.raises(FileNotFoundError):
        data.Dataset(Path("non/existent/manifest.csv"))


@pytest.fixture(scope="module")
def salami_dataset() -> data.Dataset:
    """Fixture to load the SALAMI dataset once for all tests in this module."""
    if not SALAMI_DATA_AVAILABLE:
        pytest.skip("SALAMI test data not available.")
    return data.Dataset(SALAMI_MANIFEST_PATH)


@pytest.mark.skipif(not SALAMI_DATA_AVAILABLE, reason="SALAMI data not available")
class TestSalamiDataset:
    """Test suite for the `Dataset` class using SALAMI data."""

    def test_init(self, salami_dataset: data.Dataset) -> None:
        """Test dataset initialization."""
        assert salami_dataset is not None
        assert len(salami_dataset.track_ids) > 0

    def test_list_tids(self, salami_dataset: data.Dataset) -> None:
        """Test track ID listing."""
        tids = salami_dataset.list_tids()
        assert isinstance(tids, list)
        assert len(tids) > 0
        assert all(isinstance(tid, str) and tid.isdigit() for tid in tids)

    def test_load_track(self, salami_dataset: data.Dataset) -> None:
        """Test single track loading."""
        tids = salami_dataset.list_tids()
        track = salami_dataset.load_track(tids[0])
        assert isinstance(track, data.Track)
        assert track.track_id == tids[0]
        assert track.get_asset("annotation", "reference") is not None

    def test_load_nonexistent_track(self, salami_dataset: data.Dataset) -> None:
        """Test loading a non-existent track raises an error."""
        with pytest.raises(ValueError):
            salami_dataset.load_track("nonexistent_id")

    def test_track_audio_and_info(self, salami_dataset: data.Dataset) -> None:
        """Test that track audio and metadata can be loaded."""
        track = salami_dataset.load_track("2")  # A track known to have data
        info = track.info
        assert "artist" in info and "title" in info and "duration" in info
        y, sr = track.load_audio()
        assert y is not None
        assert sr > 0

    def test_load_audio_file_not_found(
        self, salami_dataset: data.Dataset, tmp_path: Path
    ) -> None:
        """Test loading audio when the file is missing."""
        track_assets = [{"track_id": "1", "asset_type": "audio", "file_path": "missing.wav"}]
        track = data.Track.from_assets_list("1", tmp_path, track_assets)
        with pytest.raises(FileNotFoundError):
            track.load_audio()

    def test_load_audio_no_asset(self, salami_dataset: data.Dataset) -> None:
        """Test loading audio when no audio asset exists."""
        track_assets = [
            {"track_id": "1", "asset_type": "annotation", "file_path": "dummy.jams"}
        ]
        track = data.Track.from_assets_list("1", Path("."), track_assets)
        with pytest.raises(ValueError, match="No audio asset found"):
            track.load_audio()
