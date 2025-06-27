"""Core data loading classes for manifest-based datasets."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jams
import librosa
import numpy as np
import pandas as pd
import requests


def _parse_jams_metadata(jams_path: Path | str) -> dict[str, Any]:
    """Load metadata from a JAMS file, returning a dictionary."""
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
        # Consider replacing with logging in a real application
        print(f"Warning: Could not parse JAMS metadata from {jams_path}: {e}")
        return {}


@dataclass
class Track:
    """A single track, providing access to its assets via a manifest row."""

    track_id: str
    manifest_row: pd.Series
    dataset: "Dataset"

    def __repr__(self) -> str:
        # Count non-null asset paths
        num_assets = self.manifest_row.filter(like="_path").notna().sum()
        return f"Track(track_id='{self.track_id}', num_assets={num_assets}, source='{self.dataset.data_source_type}')"

    @property
    def info(self) -> dict[str, Any]:
        """A dictionary of essential track information and asset paths."""
        # The manifest row already contains all the path information.
        # We just need to resolve local paths against the dataset root.
        info_dict = self.manifest_row.to_dict()

        if self.dataset.data_source_type == "local":
            for key, value in info_dict.items():
                if key.endswith("_path") and pd.notna(value):
                    # Convert relative path string to absolute Path object
                    info_dict[key] = self.dataset.dataset_root / value

        # Add JAMS metadata if a reference annotation path exists
        jams_path_key = "annotation_reference_path"
        if jams_path := info_dict.get(jams_path_key):
            if pd.notna(jams_path):
                info_dict.update(_parse_jams_metadata(jams_path))

        return info_dict

    def load_audio(self) -> tuple[np.ndarray | None, float | None]:
        """Loads the audio waveform and sample rate for this track."""
        audio_path = self.info.get("audio_mp3_path")

        if not audio_path or pd.isna(audio_path):
            print(f"Warning: No audio asset found for track {self.track_id}.")
            return None, None

        try:
            # Check if path is a URL (string starting with http)
            if isinstance(audio_path, str) and audio_path.startswith("http"):
                response = requests.get(audio_path)
                response.raise_for_status()
                y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)
            # Check if it's a Path object that exists
            elif isinstance(audio_path, Path) and audio_path.exists():
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            else:
                print(f"Warning: Audio file not found at: {audio_path}")
                return None, None
            return y, sr
        except Exception as e:
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            return None, None


class Dataset:
    """A manifest-based dataset for local or cloud data sources.

    Parameters
    ----------
    manifest_path : Path | str
        Path or URL to the dataset's manifest CSV file.
    data_source_type : str, optional
        The type of data source, either 'local' or 'cloud'.
        This is inferred from the manifest_path but can be specified.
    """

    manifest: pd.DataFrame
    track_ids: list[str]
    data_source_type: str
    dataset_root: Path

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = manifest_path
        is_url = isinstance(manifest_path, str) and manifest_path.startswith("http")

        if is_url:
            self.data_source_type = "cloud"
            # For cloud, dataset_root is not a local path, so we don't set it.
            # URLs in the manifest are expected to be absolute.
            try:
                response = requests.get(cast(str, manifest_path))
                response.raise_for_status()
                self.manifest = pd.read_csv(io.StringIO(response.text))
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Failed to fetch cloud manifest from {manifest_path}: {e}") from e
        else:
            self.data_source_type = "local"
            manifest_path_obj = Path(manifest_path)
            if not manifest_path_obj.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path_obj}")
            self.dataset_root = manifest_path_obj.parent
            self.manifest = pd.read_csv(manifest_path_obj)

        if "track_id" not in self.manifest.columns:
            raise ValueError("Manifest must contain a 'track_id' column.")

        # Ensure track_id is string type for consistency
        self.manifest["track_id"] = self.manifest["track_id"].astype(str)

        # Store sorted unique track IDs
        unique_tids = self.manifest["track_id"].unique()
        try:
            # Try sorting numerically, but fall back to lexical sort on error
            self.track_ids = sorted(unique_tids, key=int)
        except (ValueError, TypeError):
            self.track_ids = sorted(unique_tids)

    def list_tids(self) -> list[str]:
        """A list of all track IDs in the dataset."""
        return self.track_ids

    def load_track(self, track_id: str) -> Track:
        """Loads a specific track by its ID."""
        if track_id not in self.track_ids:
            raise ValueError(f"Track ID '{track_id}' not found in manifest.")

        track_manifest_row = self.manifest[self.manifest["track_id"] == track_id]
        if track_manifest_row.empty:
            raise ValueError(f"No manifest entry found for track ID '{track_id}'.")

        return Track(track_id=track_id, manifest_row=track_manifest_row.iloc[0], dataset=self)
