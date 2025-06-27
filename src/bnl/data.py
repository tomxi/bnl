"""Core data loading classes for manifest-based datasets."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Union  # Union was missing for Path | str
from urllib.parse import urlparse

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
    manifest_row: pd.Series  # Stores the boolean flags for this track's assets
    dataset: "Dataset"  # Reference to the parent dataset to access source type, root path/URL

    def __post_init__(self) -> None:
        """Initializes cache attributes after the object is created."""
        self._info_cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        # Count how many 'has_*' columns are True
        # Ensure the boolean columns are treated as bool for sum, then cast to int
        num_assets = int(self.manifest_row.filter(like="has_").astype(bool).sum())
        return (
            f"Track(track_id='{self.track_id}', num_assets={num_assets}, "
            f"source='{self.dataset.data_source_type}')"
        )

    @property
    def info(self) -> dict[str, Any]:
        """A cached dictionary of essential track information, including paths/URLs and metadata."""
        if self._info_cache is not None:
            return self._info_cache

        info: dict[str, Any] = {"track_id": self.track_id}

        # Check if manifest uses boolean flags (has_*) or direct paths (*_path)
        has_boolean_format = any(
            col.startswith("has_") for col in self.manifest_row.index
        )

        if has_boolean_format:
            # Legacy format: Reconstruct paths/URLs based on boolean flags
            for col_name, has_asset in self.manifest_row.items():
                if col_name.startswith("has_") and has_asset:
                    # e.g., has_audio_mp3 -> audio, mp3
                    parts = col_name.replace("has_", "").split("_", 1)
                    asset_type = parts[0]
                    asset_subtype = parts[1] if len(parts) > 1 else None

                    path_or_url = self._reconstruct_path(asset_type, asset_subtype)

                    # Store the path/URL in info dict, e.g., info['audio_mp3_path'] or
                    # info['annotation_reference_path']
                    info_key_suffix = (
                        f"_{asset_subtype}_path" if asset_subtype else "_path"
                    )
                    info[f"{asset_type}{info_key_suffix}"] = path_or_url
        else:
            # Direct path format: Copy all path columns directly
            for col_name, path_value in self.manifest_row.items():
                if col_name.endswith("_path") and path_value is not None:
                    info[col_name] = path_value

        # Add JAMS metadata if reference annotation exists
        jams_info_key = "annotation_reference_path"  # Boolean format
        if jams_path_or_url := info.get(jams_info_key):
            info.update(_parse_jams_metadata(jams_path_or_url))

        # Also check for direct path manifest format
        elif "annotation_ref_path" in self.manifest_row:
            ref_path = self.manifest_row["annotation_ref_path"]
            if ref_path is not None:
                info.update(_parse_jams_metadata(ref_path))

        self._info_cache = info
        return self._info_cache

    def _reconstruct_path(
        self, asset_type: str, asset_subtype: str | None
    ) -> Path | str:
        """Reconstructs the full path or URL for an asset."""
        # This logic needs to align with manifest builders and storage layout.
        if self.dataset.data_source_type == "local":
            return self._reconstruct_local_path(asset_type, asset_subtype)
        elif self.dataset.data_source_type == "cloud":
            return self._reconstruct_cloud_url(asset_type, asset_subtype)
        else:
            raise ValueError(
                f"Invalid data_source_type: {self.dataset.data_source_type}"
            )

    def _reconstruct_local_path(
        self, asset_type: str, asset_subtype: str | None
    ) -> Path:
        """Helper to reconstruct local file paths."""
        base_path = cast(Path, self.dataset.dataset_root)
        # Example local structure:
        # <dataset_root>/audio/<track_id>/audio.mp3
        # <dataset_root>/jams/<track_id>.jams
        if asset_type == "audio":
            # Assumes audio files are <track_id>/audio.<subtype> in 'audio' folder
            return (
                base_path
                / asset_type
                / self.track_id
                / f"audio.{asset_subtype or 'mp3'}"
            )
        elif asset_type == "annotation":
            if asset_subtype == "reference":
                return base_path / "jams" / f"{self.track_id}.jams"
            # Placeholder for other local annotation types
            return base_path / asset_type / f"{self.track_id}.{asset_subtype}.json"
        else:
            # Generic fallback
            return (
                base_path
                / asset_type
                / self.track_id
                / f"asset.{asset_subtype or 'dat'}"
            )

    def _reconstruct_cloud_url(self, asset_type: str, asset_subtype: str | None) -> str:
        """Helper to reconstruct cloud URLs."""
        base_url = cast(str, self.dataset.base_url)
        # Example R2 structure from build_cloud_manifest.py:
        # {base_url}/slm-dataset/{track_id}/audio.mp3
        # {base_url}/ref-jams/{track_id}.jams
        # {base_url}/adobe21-est/.../{track_id}.mp3.msdclasscsnmagic.json
        if asset_type == "audio" and asset_subtype == "mp3":
            return f"{base_url}/slm-dataset/{self.track_id}/audio.mp3"
        elif asset_type == "annotation" and asset_subtype in ["ref", "reference"]:
            return f"{base_url}/ref-jams/{self.track_id}.jams"
        elif asset_type == "annotation" and asset_subtype == "adobe-mu1gamma1":
            return (
                f"{base_url}/adobe21-est/def_mu_0.1_gamma_0.1/"
                f"{self.track_id}.mp3.msdclasscsnmagic.json"
            )
        elif asset_type == "annotation" and asset_subtype == "adobe-mu5gamma5":
            return (
                f"{base_url}/adobe21-est/def_mu_0.5_gamma_0.5/"
                f"{self.track_id}.mp3.msdclasscsnmagic.json"
            )
        elif asset_type == "annotation" and asset_subtype == "adobe-mu1gamma9":
            return (
                f"{base_url}/adobe21-est/def_mu_0.1_gamma_0.9/"
                f"{self.track_id}.mp3.msdclasscsnmagic.json"
            )
        else:
            raise ValueError(
                f"Unknown cloud asset structure for type '{asset_type}', subtype '{asset_subtype}'"
            )

    def load_audio(self) -> tuple[np.ndarray | None, float | None]:
        """Loads the audio waveform and sample rate for this track."""
        # Find the audio asset path from the info property
        # This assumes info property correctly reconstructs paths like 'audio_mp3_path'
        audio_path_key = None
        if self.manifest_row.get("has_audio_mp3", False):  # Prioritize mp3
            audio_path_key = "audio_mp3_path"
        else:  # Fallback to any other audio type if mp3 not present
            for key in self.info.keys():
                if key.startswith("audio_") and key.endswith("_path"):
                    audio_path_key = key
                    break

        if not audio_path_key or not self.info.get(audio_path_key):
            print(
                f"Warning: No audio asset found or path not reconstructed for track "
                f"{self.track_id}."
            )
            return None, None

        audio_path = self.info[audio_path_key]

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

    # TODO: Add similar load_annotation(subtype) method
    # def load_reference_annotation(self) -> Optional[jams.JAMS]:
    #     ref_jams_path_key = "annotation_reference_path"
    #     if not self.info.get(ref_jams_path_key):
    #         print(f"Warning: No reference annotation found for track {self.track_id}")
    #         return None

    #     jams_path = self.info[ref_jams_path_key]
    #     try:
    #         if isinstance(jams_path, str) and jams_path.startswith("http"):
    #             response = requests.get(jams_path)
    #             response.raise_for_status()
    #             jam = jams.load(io.StringIO(response.text))
    #         elif Path(jams_path).exists():
    #             jam = jams.load(str(jams_path))
    #         else:
    #             return None
    #         return jam
    #     except Exception as e:
    #         print(f"Warning: Failed to load JAMS from {jams_path}: {e}")
    #         return None


class Dataset:
    """A manifest-based dataset.

    Parameters
    ----------
    manifest_path : Path | str
        The path to the dataset's `metadata.csv` manifest file.
        Can be a local file path or a URL for cloud-based manifests.
    """

    dataset_root: Path | str  # Explicitly type hint at class level
    base_url: str | None  # Explicitly type hint at class level
    manifest: pd.DataFrame
    track_ids: list[str]
    is_cloud_manifest: bool

    def __init__(
        self,
        manifest_path: Path | str,
        data_source_type: str = "cloud",
        cloud_base_url: str | None = None,
    ):
        self.manifest_path = manifest_path
        self.data_source_type = data_source_type  # 'local' or 'cloud'

        if self.data_source_type == "cloud":
            self.is_cloud_manifest = urlparse(str(manifest_path)).scheme in (
                "http",
                "https",
            )
            if not self.is_cloud_manifest:
                raise ValueError(
                    "For cloud data_source_type, manifest_path must be a URL."
                )
            # Base URL for reconstructing asset URLs, e.g., R2 public URL
            self.base_url = cloud_base_url or str(manifest_path).rsplit("/", 1)[0]
            self.dataset_root = self.base_url  # For cloud, dataset_root is the base URL
            try:
                response = requests.get(cast(str, manifest_path))
                response.raise_for_status()
                # Manifest is already boolean, no reshaping needed like _reshape_cloud_manifest
                self.manifest = pd.read_csv(io.StringIO(response.text))
            except requests.exceptions.RequestException as e:
                raise ConnectionError(
                    f"Failed to fetch cloud manifest from {manifest_path}: {e}"
                ) from e
        elif self.data_source_type == "local":
            self.is_cloud_manifest = False
            manifest_path_obj = Path(manifest_path)
            if not manifest_path_obj.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path_obj}")
            self.dataset_root = (
                manifest_path_obj.parent
            )  # Root directory for local files
            self.base_url = None  # No base_url for local data
            self.manifest = pd.read_csv(manifest_path_obj)  # Assumes boolean manifest
        else:
            raise ValueError(
                f"Unsupported data_source_type: {data_source_type}. Must be 'local' or 'cloud'."
            )

        if "track_id" in self.manifest.columns:
            self.manifest["track_id"] = self.manifest["track_id"].astype(str)
        else:
            raise ValueError("Manifest must contain a 'track_id' column.")

        unique_tids = self.manifest["track_id"].unique()
        try:
            # Try sorting numerically, but fall back to lexical sort if error
            self.track_ids = sorted(unique_tids, key=lambda x: int(x))
        except (ValueError, TypeError):
            self.track_ids = sorted(unique_tids)

    # def _reshape_cloud_manifest(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Transforms a wide cloud manifest into a long format.
    #     NOTE: This is no longer needed as manifests are expected to be boolean.
    #     """
    #     id_vars = [col for col in df.columns if not col.endswith("_path")]
    #     path_cols = [col for col in df.columns if col.endswith("_path")]

    #     if not path_cols:
    #         return df  # Return original df if no path columns found

    #     long_df = df.melt(
    #         id_vars=id_vars,
    #         value_vars=path_cols,
    #         var_name="asset_key",
    #         value_name="file_path",
    #     ).dropna(subset=["file_path"])

    #     def parse_asset_key(key: str) -> tuple[str, str]:
    #         key_no_path = key.removesuffix("_path")
    #         parts = key_no_path.split("_", 1)
    #         asset_type = parts[0]
    #         subtype = parts[1] if len(parts) > 1 else "default"

    #         if asset_type == "audio":
    #             subtype = "default"
    #         elif subtype == "ref":
    #             subtype = "reference"
    #         return asset_type, subtype

    #     parsed_keys = long_df["asset_key"].apply(parse_asset_key)
    #     long_df[["asset_type", "asset_subtype"]] = pd.DataFrame(
    #         parsed_keys.tolist(), index=long_df.index
    #     )

    #     return long_df.drop(columns=["asset_key"])

    def list_tids(self) -> list[str]:
        """A list of all track IDs in the dataset."""
        return self.track_ids

    def load_track(self, track_id: str) -> Track:
        """Loads a specific track by its ID."""
        if track_id not in self.track_ids:
            raise ValueError(f"Track ID '{track_id}' not found in manifest.")

        # Get the row from the manifest for this track_id
        track_manifest_row = self.manifest[self.manifest["track_id"] == track_id]
        if track_manifest_row.empty:
            # This should not happen if track_id is in self.track_ids
            raise ValueError(f"No manifest entry found for track ID '{track_id}'.")

        # Pass the single row (as a Series) to the Track constructor
        return Track(
            track_id=track_id, manifest_row=track_manifest_row.iloc[0], dataset=self
        )
