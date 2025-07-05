"""Edge cases for data loading, especially from JAMS files."""

import io
from unittest.mock import Mock, patch

import jams
import pandas as pd
import pytest

from bnl.data import Track


@pytest.fixture
def empty_track(tmp_path):
    """Provides a Track instance with no real data, for mocking purposes."""
    dataset = Mock()
    dataset.data_location = "local"
    dataset.dataset_root = tmp_path
    manifest_row = pd.Series({"track_id": "test"})
    return Track(track_id="test", manifest_row=manifest_row, dataset=dataset)


def test_select_jams_annotation_multiple_matches(empty_track):
    """
    Tests that a warning is printed when `_select_jams_annotation` finds
    multiple matches for a string-based `annotation_id`.
    """
    jam = jams.JAMS()
    jam.annotations.append(jams.Annotation(namespace="ns1"))
    jam.annotations.append(jams.Annotation(namespace="ns1"))

    with patch("builtins.print") as mock_print:
        selected = empty_track._select_jams_annotation(jam, "ns1", "test.jams")
        assert selected == jam.annotations[0]
        mock_print.assert_called_with("Warning: Multiple matches for 'ns1' in test.jams. Using first.")


def test_find_default_jams_annotation_logic(empty_track):
    """
    Tests the fallback logic of `_find_default_jams_annotation`.
    """
    # 1. Test that "multi_segment" is preferred
    jam_multi = jams.JAMS()
    jam_multi.annotations.append(jams.Annotation("segment_open"))
    jam_multi.annotations.append(jams.Annotation("multi_segment"))
    assert empty_track._find_default_jams_annotation(jam_multi, "").namespace == "multi_segment"

    # 2. Test that "segment_open" is the next fallback
    jam_open = jams.JAMS()
    jam_open.annotations.append(jams.Annotation("segment_other"))
    jam_open.annotations.append(jams.Annotation("segment_open"))
    assert empty_track._find_default_jams_annotation(jam_open, "").namespace == "segment_open"

    # 3. Test that multiple matches for a default type prints a warning
    jam_warn = jams.JAMS()
    jam_warn.annotations.append(jams.Annotation("multi_segment"))
    jam_warn.annotations.append(jams.Annotation("multi_segment"))
    with patch("builtins.print") as mock_print:
        empty_track._find_default_jams_annotation(jam_warn, "warn.jams")
        mock_print.assert_called_with("Warning: Multiple 'multi_segment' in warn.jams. Using first.")

    # 4. Test that it raises an error when no default types are found
    jam_fail = jams.JAMS()
    jam_fail.annotations.append(jams.Annotation("other"))
    with pytest.raises(ValueError, match="Cannot auto-load"):
        empty_track._find_default_jams_annotation(jam_fail, "fail.jams")


def test_load_invalid_json(empty_track):
    """Tests that loading malformed JSON data raises a ValueError."""
    # Patch the annotations property to avoid issues with the mock manifest
    with (
        patch.object(Track, "annotations", new={"dummy.json": "dummy.json"}),
        patch.object(empty_track, "_fetch_content", return_value=io.StringIO("{invalid_json")),
    ):
        with pytest.raises(ValueError, match="Invalid JSON"):
            empty_track.load_annotation("dummy.json")
