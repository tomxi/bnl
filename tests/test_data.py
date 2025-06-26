"""Tests for BNL data loading functionality."""

import tempfile
from pathlib import Path

import pytest

import bnl.data as data
from bnl.data.base import DatasetConfig, get_config, set_config
from bnl.data.salami import find_audio_file, list_tids


def test_config():
    """Test basic configuration functionality."""
    config = data.get_config()
    assert config.data_root == Path.home() / "data"
    assert config.salami_annotations_dir == Path.home() / "data" / "salami-jams"
    assert config.salami_audio_dir == Path.home() / "data" / "salami" / "audio"


def test_list_tids():
    """Test track ID listing."""
    tids = data.slm.list_tids()
    assert isinstance(tids, list)
    # If SALAMI data is available, should find tracks
    if len(tids) > 0:
        assert all(tid.isdigit() for tid in tids)
        # Should be sorted numerically
        assert tids == sorted(tids, key=int)


@pytest.mark.skipif(
    not (Path.home() / "data" / "salami-jams").exists(),
    reason="SALAMI data not available",
)
def test_load_track():
    """Test single track loading (requires SALAMI data)."""
    tids = data.slm.list_tids()
    if len(tids) == 0:
        pytest.skip("No SALAMI tracks available")

    track = data.slm.load_track(tids[0])

    # Basic properties
    assert track.track_id == tids[0]
    assert track.jams_path is not None
    assert track.jams_path.exists()

    # Info should be accessible
    info = track.info
    assert isinstance(info, dict)
    assert "artist" in info
    assert "title" in info
    assert "duration" in info

    # JAMS object should be accessible
    jams_obj = track.jams
    assert hasattr(jams_obj, "file_metadata")


@pytest.mark.skipif(
    not (Path.home() / "data" / "salami-jams").exists(),
    reason="SALAMI data not available",
)
def test_load_tracks():
    """Test batch track loading (requires SALAMI data)."""
    tids = data.slm.list_tids()[:3]  # Test with first 3 tracks
    if len(tids) == 0:
        pytest.skip("No SALAMI tracks available")

    tracks = data.slm.load_tracks(tids)

    assert len(tracks) <= len(tids)  # May be less if some fail to load
    assert all(isinstance(track, data.slm.Track) for track in tracks)
    assert all(track.track_id in tids for track in tracks)


def test_track_repr():
    """Test track representation doesn't crash."""
    # Test with minimal track object
    track = data.slm.Track(track_id="999")
    repr_str = repr(track)
    assert "SALAMI" in repr_str
    assert "999" in repr_str


def test_nonexistent_track():
    """Test loading nonexistent track raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        data.slm.load_track("999999")  # Very unlikely to exist


def test_module_structure():
    """Test that the simplified module structure works."""
    # Should have simplified namespace
    assert hasattr(data, "slm")
    assert hasattr(data, "get_config")
    assert hasattr(data, "set_config")

    # SALAMI functions should be in slm namespace
    assert hasattr(data.slm, "load_track")
    assert hasattr(data.slm, "load_tracks")
    assert hasattr(data.slm, "list_tids")
    assert hasattr(data.slm, "Track")


def test_salami_edge_cases():
    """Test edge cases for SALAMI module coverage."""
    # Test load_tracks with non-existent track to trigger FileNotFoundError handling
    tracks = data.slm.load_tracks(["999999"])  # Very unlikely to exist
    assert tracks == []  # Should return empty list after catching error

    # Test find_audio_file with non-existent track_id to cover line 26
    fake_dir = Path("/non/existent/directory")
    result = find_audio_file("nonexistent", fake_dir)
    assert result is None


def test_dataset_config_and_missing_paths():
    """Test custom dataset config and handling of missing files/directories."""
    original_config = get_config()

    # Create config with custom non-existent paths (covers path conversion lines)
    custom_config = DatasetConfig(
        data_root="/tmp/nonexistent_data",
        salami_annotations_dir="/tmp/nonexistent_jams",
        salami_audio_dir="/tmp/nonexistent_audio",
        adobe_estimations_dir="/tmp/nonexistent_adobe",
    )

    # Verify Path objects were created from string inputs
    assert isinstance(custom_config.data_root, Path)
    assert isinstance(custom_config.salami_annotations_dir, Path)
    assert isinstance(custom_config.salami_audio_dir, Path)
    assert isinstance(custom_config.adobe_estimations_dir, Path)

    # Set the custom config globally
    set_config(custom_config)
    assert get_config() == custom_config

    # Test missing directory scenarios
    tids = list_tids()  # Should handle missing annotations_dir gracefully
    assert tids == []

    result = find_audio_file(
        "test", Path("/tmp/nonexistent")
    )  # Missing track directory
    assert result is None

    # Restore original config
    set_config(original_config)


def test_find_audio_file_branches():
    """Test find_audio_file branches."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        assert find_audio_file("x", p) is None  # no dir
        (p / "y").mkdir()
        assert find_audio_file("y", p) is None  # no audio
        (p / "y" / "audio.mp3").touch()
        assert find_audio_file("y", p) is not None  # has audio
