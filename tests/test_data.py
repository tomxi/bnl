"""Tests for the data loading module."""

from pathlib import Path

import pytest

from bnl import core, data

# Use a real track from the test fixtures
TEST_TID = "8"


@pytest.fixture(scope="module")
def dataset() -> data.Dataset:
    """Provides a Dataset instance for testing."""
    return data.Dataset(Path(__file__).parent / "fixtures" / "test-manifest.csv")


@pytest.fixture(scope="module")
def dataset_cloud() -> data.Dataset:
    return data.Dataset()


@pytest.mark.parametrize("ds_fixture", ["dataset", "dataset_cloud"])
def test_dataset_loading(ds_fixture, request):
    """Tests basic dataset and track loading."""
    ds = request.getfixturevalue(ds_fixture)
    assert ds.data_location in ["local", "cloud"]
    assert len(ds) > 0
    assert isinstance(ds.track_ids, list)
    assert all(isinstance(tid, str) for tid in ds.track_ids)
    track = ds[TEST_TID]
    assert track.track_id == TEST_TID
    assert len(track.info) > 0


def test_dataset_manifest_not_found():
    """Test that a FileNotFoundError is raised for a non-existent manifest."""
    with pytest.raises(FileNotFoundError):
        data.Dataset(manifest_path="/non/existent/path/manifest.csv")


@pytest.mark.parametrize("ds_fixture", ["dataset", "dataset_cloud"])
def test_track_properties_and_errors(ds_fixture, request):
    """Test track properties and error handling for loading annotations."""
    ds = request.getfixturevalue(ds_fixture)
    track = ds[TEST_TID]

    # Test __repr__
    assert f"Track(track_id='{TEST_TID}'" in repr(track)
    assert "num_assets=" in repr(track)

    # Test for annotation type that doesn't exist
    with pytest.raises(ValueError, match="not available"):
        track.load_annotation("non_existent_type")

    # Test for file type that is not supported
    track.info["annotation_fake_type_path"] = "/fake/path/file.txt"
    with pytest.raises(NotImplementedError, match="Unsupported file type"):
        track.load_annotation("fake_type")


def test_load_cloud_est(dataset: data.Dataset, dataset_cloud: data.Dataset):
    test_track_est = dataset[TEST_TID].load_annotation("adobe-mu1gamma1")
    cloud_track_est = dataset_cloud[TEST_TID].load_annotation("adobe-mu1gamma1")

    assert isinstance(test_track_est, core.MultiSegment)
    assert len(test_track_est.layers) >= 1
    assert test_track_est == cloud_track_est


def test_load_cloud_ref(dataset: data.Dataset, dataset_cloud: data.Dataset):
    test_track_ref = dataset[TEST_TID].load_annotation("reference")
    cloud_track_ref = dataset_cloud[TEST_TID].load_annotation("reference")

    assert isinstance(test_track_ref, core.MultiSegment)
    assert len(test_track_ref.layers) == 2
    assert test_track_ref == cloud_track_ref


@pytest.mark.parametrize("ds_fixture", ["dataset", "dataset_cloud"])
def test_load_reference(ds_fixture, request):
    ds = request.getfixturevalue(ds_fixture)
    track = ds[TEST_TID]
    for ref_id in track.refs:
        assert isinstance(track.refs[ref_id], core.MultiSegment)
        assert len(track.refs[ref_id].layers) == 2


@pytest.mark.parametrize("ds_fixture", ["dataset", "dataset_cloud"])
def test_load_est(ds_fixture, request):
    ds = request.getfixturevalue(ds_fixture)
    track = ds[TEST_TID]
    for est_id in track.ests:
        assert isinstance(track.ests[est_id], core.MultiSegment)
        assert len(track.ests[est_id].layers) >= 1


@pytest.mark.parametrize("ds_fixture", ["dataset", "dataset_cloud"])
def test_dataset_iter(ds_fixture, request):
    ds = request.getfixturevalue(ds_fixture)
    for track in ds:
        assert isinstance(track, data.Track)
        with pytest.raises(
            ValueError,
            match="Annotation type 'non_existent_type' not available for this track.",
        ):
            track.load_annotation("non_existent_type")


@pytest.mark.parametrize("ds_fixture", ["dataset", "dataset_cloud"])
def test_dataset_bad_id(ds_fixture, request):
    ds = request.getfixturevalue(ds_fixture)
    with pytest.raises(ValueError, match="Track ID 'non_existent_id' not found in manifest."):
        ds["non_existent_id"]
