"""Core data loading classes for manifest-based datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jams
import librosa
import numpy as np
import pandas as pd


def _parse_jams_metadata(jams_path: Path) -> dict[str, Any]:
    """Load metadata from a JAMS file, returning a dictionary.

    Handles all error cases and returns empty dict on failure.
    """
    if not jams_path.exists():
        return {}

    try:
        jam = jams.load(str(jams_path))
        meta = jam.file_metadata
        return {"title": meta.title, "artist": meta.artist, "duration": meta.duration}
    except Exception as e:
        print(f"Warning: Could not parse JAMS metadata from {jams_path}: {e}")
        return {}


@dataclass
class Track:
    """A single track and its associated data assets.

    Parameters
    ----------
    track_id : str
        The unique identifier for the track.
    dataset_root : Path
        The root path of the dataset, used to resolve relative file paths.
    assets : dict
        A dictionary mapping asset identifiers (type, subtype) to their
        metadata.
    """

    track_id: str
    dataset_root: Path
    assets: dict[tuple[str, str], dict[str, Any]]

    def __post_init__(self) -> None:
        """Initializes cache attributes after the object is created."""
        self._info_cache: dict[str, Any] | None = None

    @classmethod
    def from_assets_list(
        cls, track_id: str, dataset_root: Path, assets: list[dict[str, Any]]
    ) -> "Track":
        """Create a `Track` from a list of asset dictionaries."""
        # Flatten to composite keys for simple O(1) access
        assets_dict = {}
        for asset in assets:
            key = (asset["asset_type"], asset.get("asset_subtype", "default"))
            assets_dict[key] = asset

        return cls(track_id, dataset_root, assets_dict)

    def __repr__(self) -> str:
        return f"Track(track_id='{self.track_id}', num_assets={len(self.assets)})"

    @property
    def info(self) -> dict[str, Any]:
        """A cached dictionary of essential track information."""
        if self._info_cache is not None:
            return self._info_cache

        info: dict[str, Any] = {"track_id": self.track_id}

        # Add audio path if it exists
        if audio_asset := self.get_asset("audio"):
            info["audio_path"] = self.resolve_path(audio_asset["file_path"])

        # Add JAMS metadata if reference annotation exists
        if jams_asset := self.get_asset("annotation", "reference"):
            jams_path = self.resolve_path(jams_asset["file_path"])
            info.update(_parse_jams_metadata(jams_path))

        self._info_cache = info
        return self._info_cache

    def resolve_path(self, relative_path: str) -> Path:
        """Resolves a relative asset path against the dataset root."""
        return self.dataset_root / relative_path

    def get_asset(
        self, asset_type: str, asset_subtype: str = "default"
    ) -> dict[str, Any] | None:
        """Get an asset by its type and subtype.

        If the exact subtype is not found, it falls back to the first asset
        of the specified `asset_type`.
        """
        # Exact match
        if asset := self.assets.get((asset_type, asset_subtype)):
            return asset

        # Fallback: find any asset of the given type if exact match fails
        for key, asset_dict in self.assets.items():
            if key[0] == asset_type:
                return asset_dict

        return None

    def load_audio(self) -> tuple[np.ndarray, float]:
        """Loads the audio waveform and sample rate for this track."""
        audio_asset = self.get_asset("audio")
        if audio_asset is None:
            raise ValueError(f"No audio asset found for track {self.track_id}")

        audio_path = self.resolve_path(audio_asset["file_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr


class Dataset:
    """A manifest-based dataset.

    Parameters
    ----------
    manifest_path : Path
        The path to the dataset's `metadata.csv` manifest file.
    """

    def __init__(self, manifest_path: Path):
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.manifest_path = manifest_path
        self.dataset_root = manifest_path.parent
        self.manifest = pd.read_csv(manifest_path)

        if "track_id" in self.manifest.columns:
            self.manifest["track_id"] = self.manifest["track_id"].astype(str)
        self.track_ids = sorted(self.manifest["track_id"].unique(), key=lambda x: int(x))

    def list_tids(self) -> list[str]:
        """A list of all track IDs in the dataset."""
        return self.track_ids

    def load_track(self, track_id: str) -> Track:
        """Loads a specific track by its ID."""
        if track_id not in self.track_ids:
            raise ValueError(f"Track ID '{track_id}' not found in manifest.")

        # Convert DataFrame rows to list of dicts
        track_assets = self.manifest[self.manifest["track_id"] == track_id]
        assets_list = cast(list[dict[str, Any]], track_assets.to_dict("records"))

        return Track.from_assets_list(track_id, self.dataset_root, assets_list)
