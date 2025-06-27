"""Core data loading classes for manifest-based datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
import io

import jams
import librosa
import numpy as np
import pandas as pd
import requests


def _parse_jams_metadata(jams_path: Path | str) -> dict[str, Any]:
    """Load metadata from a JAMS file, returning a dictionary.

    Handles all error cases and returns empty dict on failure.
    """
    try:
        if isinstance(jams_path, str) and jams_path.startswith("http"):
            response = requests.get(jams_path)
            response.raise_for_status()
            jam = jams.load(io.StringIO(response.text))
        elif Path(jams_path).exists():
            jam = jams.load(str(jams_path))
        else:
            return {}

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
    dataset_root : Path | str
        The root path of the dataset, used to resolve relative file paths.
        Can be a local Path or a URL base for cloud datasets.
    assets : dict
        A dictionary mapping asset identifiers (type, subtype) to their
        metadata.
    """

    track_id: str
    dataset_root: Path | str
    assets: dict[tuple[str, str], dict[str, Any]]

    def __post_init__(self) -> None:
        """Initializes cache attributes after the object is created."""
        self._info_cache: dict[str, Any] | None = None

    @classmethod
    def from_assets_list(
        cls, track_id: str, dataset_root: Path | str, assets: list[dict[str, Any]]
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

        for asset_key, asset_data in self.assets.items():
            asset_type, asset_subtype = asset_key
            path_key = (
                f"{asset_type}_path"
                if asset_subtype == "default"
                else f"{asset_type}_{asset_subtype}_path"
            )
            info[path_key] = self.resolve_path(asset_data["file_path"])

        # Add JAMS metadata if reference annotation exists
        if jams_path := info.get("annotation_reference_path"):
            info.update(_parse_jams_metadata(jams_path))

        self._info_cache = info
        return self._info_cache

    def resolve_path(self, relative_path: str) -> Path | str:
        """Resolves a relative asset path against the dataset root."""
        if isinstance(self.dataset_root, str):
            # For URL-based datasets, return as-is or join with base URL
            return relative_path
        else:
            # For local datasets, resolve against the root path
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

    def load_audio(self) -> tuple[np.ndarray | None, float | None]:
        """Loads the audio waveform and sample rate for this track."""
        audio_asset = self.get_asset("audio")
        if not audio_asset:
            return None, None

        audio_path = self.resolve_path(audio_asset["file_path"])

        try:
            if isinstance(audio_path, str) and audio_path.startswith("http"):
                response = requests.get(audio_path)
                response.raise_for_status()
                y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)
            elif Path(audio_path).exists():
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            else:
                print(f"Warning: Audio file not found at: {audio_path}")
                return None, None
            return y, sr
        except Exception as e:
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            return None, None


class Dataset:
    """A manifest-based dataset.

    Parameters
    ----------
    manifest_path : Path | str
        The path to the dataset's `metadata.csv` manifest file.
        Can be a local file path or a URL for cloud-based manifests.
    """

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = manifest_path
        self.is_cloud = isinstance(manifest_path, str) and urlparse(
            manifest_path
        ).scheme in ("http", "https")

        if self.is_cloud:
            self.dataset_root = str(manifest_path).rsplit("/", 1)[0]
            try:
                response = requests.get(cast(str, manifest_path))
                response.raise_for_status()
                manifest_df = pd.read_csv(io.StringIO(response.text))
                self.manifest = self._reshape_cloud_manifest(manifest_df)
            except requests.exceptions.RequestException as e:
                raise ConnectionError(
                    f"Failed to fetch cloud manifest from {manifest_path}: {e}"
                )
        else:
            manifest_path = Path(manifest_path)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.dataset_root = manifest_path.parent
            self.manifest = pd.read_csv(manifest_path)

        if "track_id" in self.manifest.columns:
            self.manifest["track_id"] = self.manifest["track_id"].astype(str)

        unique_tids = self.manifest["track_id"].unique()
        try:
            # Try sorting numerically, but fall back to lexical sort if error
            self.track_ids = sorted(unique_tids, key=lambda x: int(x))
        except (ValueError, TypeError):
            self.track_ids = sorted(unique_tids)

    def _reshape_cloud_manifest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a wide cloud manifest into a long format."""
        id_vars = [col for col in df.columns if not col.endswith("_path")]
        path_cols = [col for col in df.columns if col.endswith("_path")]

        if not path_cols:
            return df  # Return original df if no path columns found

        long_df = df.melt(
            id_vars=id_vars,
            value_vars=path_cols,
            var_name="asset_key",
            value_name="file_path",
        ).dropna(subset=["file_path"])

        def parse_asset_key(key: str) -> tuple[str, str]:
            key_no_path = key.removesuffix("_path")
            parts = key_no_path.split("_", 1)
            asset_type = parts[0]
            subtype = parts[1] if len(parts) > 1 else "default"

            if asset_type == "audio":
                subtype = "default"
            elif subtype == "ref":
                subtype = "reference"
            return asset_type, subtype

        parsed_keys = long_df["asset_key"].apply(parse_asset_key)
        long_df[["asset_type", "asset_subtype"]] = pd.DataFrame(
            parsed_keys.tolist(), index=long_df.index
        )

        return long_df.drop(columns=["asset_key"])

    def list_tids(self) -> list[str]:
        """A list of all track IDs in the dataset."""
        return self.track_ids

    def load_track(self, track_id: str) -> Track:
        """Loads a specific track by its ID."""
        if track_id not in self.track_ids:
            raise ValueError(f"Track ID '{track_id}' not found in manifest.")

        track_assets = self.manifest[self.manifest["track_id"] == track_id]
        if track_assets.empty:
            raise ValueError(f"No assets found for track ID '{track_id}'.")

        assets_list = cast(list[dict[str, Any]], track_assets.to_dict("records"))
        return Track.from_assets_list(track_id, self.dataset_root, assets_list)
