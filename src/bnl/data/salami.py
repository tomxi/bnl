"""SALAMI dataset loader and utilities.

This module provides functions to load and work with the SALAMI
(Structural Analysis of Large Amounts of Music Information) dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .base import BaseTrack, get_config


@dataclass
class Track(BaseTrack):
    """A SALAMI track with annotations and metadata."""

    def __repr__(self) -> str:
        return f"SALAMI {super().__repr__()}"


def find_audio_file(track_id: str, audio_dir: Path) -> Optional[Path]:
    """Find the audio file for a given track ID."""
    track_dir = audio_dir / track_id
    if not track_dir.exists():
        return None

    # Common audio extensions
    audio_extensions = [".mp3", ".wav", ".flac", ".m4a"]

    for ext in audio_extensions:
        audio_file = track_dir / f"audio{ext}"
        if audio_file.exists():
            return audio_file

    return None


def load_track(track_id: Union[str, int]) -> Track:
    """Load a single SALAMI track.

    Parameters
    ----------
    track_id : str or int
        SALAMI track identifier.

    Returns
    -------
    Track
        Loaded track with metadata and file paths.

    Raises
    ------
    FileNotFoundError
        If the JAMS file for the track doesn't exist.
    """
    track_id = str(track_id)
    config = get_config()

    # Find JAMS file
    jams_path = config.salami_annotations_dir / f"{track_id}.jams"
    if not jams_path.exists():
        raise FileNotFoundError(f"JAMS file not found: {jams_path}")

    # Find audio file
    audio_path = find_audio_file(track_id, config.salami_audio_dir)

    return Track(track_id=track_id, jams_path=jams_path, audio_path=audio_path)


def load_tracks(track_ids: List[Union[str, int]]) -> List[Track]:
    """Load multiple SALAMI tracks.

    Parameters
    ----------
    track_ids : list of str or int
        List of SALAMI track identifiers.

    Returns
    -------
    list of Track
        List of loaded tracks. Tracks that fail to load are skipped
        with a warning message.
    """
    tracks = []

    for track_id in track_ids:
        try:
            track = load_track(track_id)
            tracks.append(track)
        except FileNotFoundError as e:
            print(f"Warning: Could not load track {track_id}: {e}")
            continue

    return tracks


def list_tids() -> List[str]:
    """List all available SALAMI track IDs.

    Returns
    -------
    list of str
        List of available track IDs, sorted numerically.
    """
    load_config = get_config()
    annotations_dir = load_config.salami_annotations_dir

    if not annotations_dir.exists():
        print(f"Warning: SALAMI annotations directory not found: {annotations_dir}")
        return []

    # Find all .jams files and extract track IDs
    track_ids = []
    for jams_file in annotations_dir.glob("*.jams"):
        track_id = jams_file.stem
        # Verify it's a numeric track ID
        if track_id.isdigit():
            track_ids.append(track_id)

    # Sort numerically
    track_ids.sort(key=int)
    return track_ids
