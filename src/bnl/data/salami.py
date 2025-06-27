"""SALAMI dataset loader and utilities.

This module provides functions to load and work with the SALAMI
(Structural Analysis of Large Amounts of Music Information) dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .base import BaseDataset, BaseTrack, get_config


@dataclass
class Track(BaseTrack):
    """A SALAMI track with annotations and metadata."""

    def __repr__(self) -> str:
        return f"SALAMI {super().__repr__()}"


class SalamiDataset(BaseDataset):
    """SALAMI dataset with manifest-based loading."""

    def load_track(self, track_id: Union[str, int]) -> Track:
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
            If required files for the track don't exist.
        ValueError
            If track_id is not found in the manifest.
        """
        track_id = str(track_id)

        # Get file paths from manifest
        audio_path = self.get_audio_path(track_id)
        jams_path = self.get_jams_path(track_id)

        # Fail fast: ensure we have at least one asset for the track
        if audio_path is None and jams_path is None:
            raise ValueError(f"Track {track_id} not found in dataset")

        return Track(track_id=track_id, jams_path=jams_path, audio_path=audio_path)

    def load_tracks(self, track_ids: List[Union[str, int]]) -> List[Track]:
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
                track = self.load_track(track_id)
                tracks.append(track)
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load track {track_id}: {e}")
                continue

        return tracks

    def list_tids(self) -> List[str]:
        """List all available SALAMI track IDs.

        Returns
        -------
        list of str
            List of available track IDs, sorted numerically.
        """
        return self.list_track_ids()


# Global dataset instance - lazy loaded
_salami_dataset = None


def _get_salami_dataset() -> SalamiDataset:
    """Get or create the global SALAMI dataset instance."""
    global _salami_dataset

    if _salami_dataset is None:
        # Try manifest-based loading first
        config = get_config()
        manifest_path = config.data_root / "salami" / "metadata.csv"

        if manifest_path.exists():
            _salami_dataset = SalamiDataset(manifest_path)
        else:
            # Fallback to legacy filesystem scanning
            # This will be deprecated once manifests are standard
            raise FileNotFoundError(
                f"SALAMI manifest not found: {manifest_path}\n"
                f"Please generate it with: python scripts/build_manifest.py {config.data_root / 'salami'}"
            )

    return _salami_dataset


# Public API functions that maintain backward compatibility
def load_track(track_id: Union[str, int]) -> Track:
    """Load a single SALAMI track."""
    return _get_salami_dataset().load_track(track_id)


def load_tracks(track_ids: List[Union[str, int]]) -> List[Track]:
    """Load multiple SALAMI tracks."""
    return _get_salami_dataset().load_tracks(track_ids)


def list_tids() -> List[str]:
    """List all available SALAMI track IDs."""
    return _get_salami_dataset().list_tids()


# Legacy functions - deprecated but maintained for compatibility
def find_audio_file(track_id: str, audio_dir: Path) -> Optional[Path]:
    """Find the audio file for a given track ID.

    DEPRECATED: Use manifest-based loading instead.
    """
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
