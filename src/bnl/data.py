"""Core data loading classes for manifest-based datasets."""

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import jams
import librosa
import numpy as np
import pandas as pd
import requests

from .core import Hierarchy


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
    manifest_row: pd.Series  # Stores the boolean flags for this track's assets
    dataset: "Dataset"  # Reference to the parent dataset

    def __post_init__(self) -> None:
        """Initializes cache attributes after the object is created."""
        self._info_cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        # Count how many 'has_*' columns are True
        # Ensure the boolean columns are treated as bool for sum, then cast to int
        has_columns = self.manifest_row.filter(like="has_").astype(bool)
        num_assets = int(has_columns.values.sum()) if len(has_columns) > 0 else 0
        return f"Track(track_id='{self.track_id}', num_assets={num_assets}, source='{self.dataset.data_location}')"

    @property
    def info(self) -> dict[str, Any]:
        """A cached dictionary of essential track information."""
        if self._info_cache is not None:
            return self._info_cache

        info: dict[str, Any] = {"track_id": self.track_id}

        # Reconstruct paths/URLs for all available assets based on boolean flags
        for col_name, has_asset in self.manifest_row.items():
            if str(col_name).startswith("has_") and has_asset:
                # e.g., has_audio_mp3 -> ('audio', 'mp3')
                parts = col_name.replace("has_", "").split("_", 1)
                asset_type = parts[0]
                asset_subtype = parts[1] if len(parts) > 1 else None

                if asset_type and asset_subtype:
                    path_or_url = self.dataset._reconstruct_path(self.track_id, asset_type, asset_subtype)
                    # e.g., info['audio_mp3_path']
                    info_key = f"{asset_type}_{asset_subtype}_path"
                    info[info_key] = path_or_url

        # Add JAMS metadata if a reference annotation exists
        if jams_path_or_url := info.get("annotation_reference_path"):
            info.update(_parse_jams_metadata(jams_path_or_url))

        self._info_cache = info
        return self._info_cache

    @property
    def has_annotations(self) -> bool:
        """Checks if the track has any associated annotations."""
        # Check for any column that starts with 'has_annotation_' and is True
        return self.manifest_row.filter(like="has_annotation_").any()

    @property
    def annotations(self) -> dict[str, str | Path]:
        """Returns a dictionary of available annotation paths."""
        ann_paths = {}
        for key, value in self.info.items():
            if key.startswith("annotation_") and key.endswith("_path"):
                # e.g., annotation_reference_path -> reference
                ann_type = key.replace("annotation_", "").replace("_path", "")
                ann_paths[ann_type] = value
        return ann_paths

    def load_audio(self) -> tuple[np.ndarray | None, float | None]:
        """Loads the audio waveform and sample rate for this track."""
        # Find the primary audio asset path from the info dictionary
        audio_path_key = None
        for key in self.info.keys():
            if key.startswith("audio_") and key.endswith("_path"):
                audio_path_key = key
                break  # Use the first audio asset found

        if not audio_path_key:
            return None, None

        audio_path = self.info[audio_path_key]

        # Expand user path if it's a local path
        expanded_audio_path = Path(audio_path).expanduser() if isinstance(audio_path, str | Path) else audio_path

        try:
            if isinstance(audio_path, str) and audio_path.startswith("http"):
                response = requests.get(audio_path)
                response.raise_for_status()
                y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)
            elif expanded_audio_path.exists():
                y, sr = librosa.load(expanded_audio_path, sr=None, mono=True)
            else:
                print(f"Warning: Audio file not found at: {audio_path}")
                return None, None
            return y, sr
        except Exception as e:
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            return None, None

    def load_hierarchy(self, annotation_type: str) -> "bnl.Hierarchy":
        """Load a hierarchical annotation as a Hierarchy object.

        Parameters
        ----------
        annotation_type : str
            Type of annotation to load (e.g., 'reference')

        Returns
        -------
        bnl.Hierarchy
            The loaded hierarchical annotation
        """
        import jams

        # Get the annotation path
        if annotation_type not in self.annotations:
            raise ValueError(f"Annotation type '{annotation_type}' not available for this track")

        annotation_path = self.annotations[annotation_type]

        # Load the JAMS file
        if self.dataset.data_location == "cloud":
            # For cloud data, download and load
            response = requests.get(str(annotation_path))
            response.raise_for_status()
            jam = jams.load(io.StringIO(response.text))
        else:
            # For local data, load directly
            jam = jams.load(str(annotation_path))

        # Find multi_segment annotation
        multi_segment_ann = None
        for ann in jam.annotations:
            if ann.namespace == "multi_segment":
                multi_segment_ann = ann
                break

        if multi_segment_ann is None:
            raise ValueError(f"No multi_segment annotation found in {annotation_path}")

        return Hierarchy.from_jams(multi_segment_ann)


class Dataset:
    """A manifest-based dataset.

    Parameters
    ----------
    manifest_path : Path | str
        The path to the dataset's `metadata.csv` manifest file.
        Can be a local file path or a URL for cloud-based manifests.
    """

    def __init__(
        self,
        manifest_path: Path | str,
    ):
        # --- Load the manifest ---
        self.manifest_path = manifest_path
        self.data_location = "cloud" if urlparse(str(manifest_path)).scheme in ("http", "https") else "local"

        if self.data_location == "local":
            # Expand user path (~ -> /home/user) before creating Path
            expanded_manifest_path = Path(manifest_path).expanduser()
            self.dataset_root: Path | str = expanded_manifest_path.parent
            self.base_url: str | None = None
        elif self.data_location == "cloud":
            self.dataset_root = self.base_url
            self.base_url = str(manifest_path).rsplit("/", 1)[0]
        else:
            raise ValueError(f"Unsupported data_location: '{self.data_location}'")

        try:
            # Use expanded path for local files, original path for URLs
            load_path = Path(manifest_path).expanduser() if self.data_location == "local" else manifest_path
            self.manifest = pd.read_csv(load_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}") from e

            # --- Standardize track_id to string ---
        if "track_id" not in self.manifest.columns:
            raise ValueError("Manifest must contain a 'track_id' column.")
        self.manifest["track_id"] = self.manifest["track_id"].astype(str)
        self.manifest.set_index("track_id", inplace=True, drop=False)

        # Sort by track_id, handling numeric vs. lexical gracefully
        self.track_ids = sorted(self.manifest["track_id"].astype(str).unique(), key=lambda x: int(x))

    def __getitem__(self, track_id: str) -> Track:
        """Load a specific track by its ID."""
        track_id = str(track_id)  # Ensure track_id is a string
        if track_id not in self.track_ids:
            raise ValueError(f"Track ID '{track_id}' not found in manifest.")
        return Track(track_id, self.manifest.loc[track_id], self)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __iter__(self) -> Iterator[Track]:
        for track_id in self.track_ids:
            yield self[track_id]

    def _reconstruct_path(self, track_id: str, asset_type: str, asset_subtype: str) -> Path | str:
        """Reconstructs the full path or URL for an asset.

        This logic must align with the conventions used by the manifest builders
        (e.g., `scripts/build_local_manifest.py`).
        """
        if self.data_location == "local":
            # --- Local Path Reconstruction ---
            # This logic mirrors the structure in `scripts/build_local_manifest.py`
            root = cast(Path, self.dataset_root)
            if asset_type == "audio":
                return root / "audio" / track_id / f"audio.{asset_subtype}"
            elif asset_type == "annotation":
                if asset_subtype == "reference":
                    return root / "jams" / f"{track_id}.jams"
                elif "adobe" in asset_subtype:
                    # e.g., adobe-mu1gamma1 -> adobe/def_mu_0.1_gamma_0.1
                    mu_gamma = asset_subtype.split("-")[1].replace("mu", "mu_").replace("gamma", "_gamma_")
                    subfolder = f"adobe/def_{mu_gamma.replace('.', '_', 1)}"
                    return root / subfolder / f"{track_id}.mp3.msdclasscsnmagic.json"
            raise ValueError(f"Unknown local asset structure for: {asset_type}/{asset_subtype}")

        elif self.data_location == "cloud":
            # --- Cloud URL Reconstruction ---
            # This logic mirrors the structure in `scripts/build_cloud_manifest.py`
            base = cast(str, self.base_url)
            if asset_type == "audio" and asset_subtype == "mp3":
                return f"{base}/slm-dataset/{track_id}/audio.mp3"
            elif asset_type == "annotation" and asset_subtype == "reference":
                return f"{base}/ref-jams/{track_id}.jams"
            elif asset_type == "annotation" and "adobe" in asset_subtype:
                # e.g., adobe-mu1gamma1 -> adobe21-est/def_mu_0.1_gamma_0.1
                mu_gamma = asset_subtype.split("-")[1].replace("mu", "mu_").replace("gamma", "_gamma_")
                subfolder = f"adobe21-est/def_{mu_gamma.replace('.', '_', 1)}"
                return f"{base}/{subfolder}/{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown asset structure for source type '{self.data_location}'")
