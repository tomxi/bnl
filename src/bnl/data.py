"""Core data loading classes for manifest-based datasets."""

from __future__ import annotations

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

from .core import Hierarchy, Segmentation


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
    dataset: Dataset  # Reference to the parent dataset

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

    def load_annotation(
        self, annotation_type: str, annotation_id: str | int | None = None
    ) -> Hierarchy | Segmentation:  # "Segmentation" forward ref for type checker
        """Load an annotation as a Hierarchy or Segmentation object.

        Parameters
        ----------
        annotation_type : str
            Type of annotation to load, corresponding to a key in `track.annotations`
            (e.g., 'reference', 'adobe-mu1gamma1'). This determines the file to load.
        annotation_id : str or int, optional
            If the annotation file (especially JAMS) contains multiple annotations,
            this specifies which one to load:

            - For JAMS: Can be an annotation's `id` (if set in JAMS)
              or `namespace`. If an integer, it's treated as an index
              into `jam.annotations`.
            - For JSON: Currently not used, but kept for API consistency.

            If None (default):

            - For JAMS: Attempts to find and load a 'multi_segment'
              annotation as Hierarchy. If not found, looks for other common
              segmentation types to load as Segmentation. If multiple
              suitable annotations are found, loads the first one.
            - For JSON: Loads the entire file as a Hierarchy.

        Returns
        -------
        bnl.Hierarchy or bnl.Segmentation
            The loaded annotation data.

        Raises
        ------
        ValueError
            If the annotation_type is not found, the file cannot be fetched,
            no suitable annotation is found in the file, or the specified
            annotation_id is not found.
        NotImplementedError
            If the file type is not supported.
        """
        # Import Segmentation locally if needed to avoid circular import at module level with core

        if annotation_type not in self.annotations:
            raise ValueError(
                f"Annotation type '{annotation_type}' not available for this track. "
                f"Available: {list(self.annotations.keys())}"
            )

        annotation_path = self.annotations[annotation_type]

        # --- File Fetching ---
        file_content_stream = self._fetch_annotation_content(annotation_path)

        # --- JAMS Loading Logic ---
        if str(annotation_path).lower().endswith(".jams"):
            return self._load_jams_annotation(file_content_stream, annotation_path, annotation_id)

        # --- JSON Loading Logic ---
        elif str(annotation_path).lower().endswith(".json"):
            return self._load_json_annotation(file_content_stream, annotation_path)

        # --- Unsupported File Type ---
        else:
            raise NotImplementedError(
                f"Unsupported annotation file type for: {annotation_path}. Only .jams and .json are supported."
            )

    def _fetch_annotation_content(self, annotation_path: str | Path) -> io.StringIO:
        """Fetches annotation content from local path or URL into a StringIO buffer."""
        try:
            if isinstance(annotation_path, str) and annotation_path.startswith("http"):
                response = requests.get(str(annotation_path))
                response.raise_for_status()
                return io.StringIO(response.text)
            elif Path(annotation_path).exists():
                with open(annotation_path, encoding="utf-8") as f:
                    return io.StringIO(f.read())
            else:
                raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch cloud annotation from {annotation_path}: {e}") from e
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise ValueError(f"Error reading annotation file {annotation_path}: {e}") from e

    def _load_jams_annotation(
        self, file_content_stream: io.StringIO, file_path_for_error_msg: str | Path, annotation_id: str | int | None
    ) -> Hierarchy | Segmentation:
        """Helper to load and process a JAMS annotation."""
        # Import Segmentation locally if needed
        import jams  # Ensure jams is imported locally if not at module level

        from .core import Segmentation

        jam = jams.load(file_content_stream)
        selected_ann: jams.Annotation | None = None

        if annotation_id is not None:
            if isinstance(annotation_id, int):
                if 0 <= annotation_id < len(jam.annotations):
                    selected_ann = jam.annotations[annotation_id]
                else:
                    raise ValueError(
                        f"Annotation index {annotation_id} out of range for {file_path_for_error_msg}. "
                        f"Found {len(jam.annotations)} annotations."
                    )
            elif isinstance(annotation_id, str):
                found_annotations = [
                    ann
                    for ann in jam.annotations
                    if ann.namespace == annotation_id or (hasattr(ann, "id") and ann.id == annotation_id)
                ]
                if found_annotations:
                    if len(found_annotations) > 1 and not any(
                        ann.id == annotation_id for ann in found_annotations if hasattr(ann, "id")
                    ):
                        print(
                            f"Warning: Multiple for namespace '{annotation_id}' in "
                            f"{file_path_for_error_msg}. Using first."
                        )
                    selected_ann = found_annotations[0]
                else:
                    raise ValueError(
                        f"No annotation found with id or namespace '{annotation_id}' in {file_path_for_error_msg}"
                    )
            else:
                raise TypeError(f"Invalid annotation_id type: {type(annotation_id)}. Must be int or str.")
        else:  # Default loading (annotation_id is None)
            multi_segment_annotations = [ann for ann in jam.annotations if ann.namespace == "multi_segment"]
            if multi_segment_annotations:
                if len(multi_segment_annotations) > 1:
                    print(f"Warning: Multiple 'multi_segment' in {file_path_for_error_msg}. Using first.")
                selected_ann = multi_segment_annotations[0]
            else:
                common_segment_namespaces = ["segment_open"]  # Define or pass as arg
                for ns_to_find in common_segment_namespaces:
                    segment_annotations_for_ns = [ann for ann in jam.annotations if ann.namespace == ns_to_find]
                    if segment_annotations_for_ns:
                        if len(segment_annotations_for_ns) > 1:
                            print(f"Warning: Multiple '{ns_to_find}' in {file_path_for_error_msg}. Using first.")
                        selected_ann = segment_annotations_for_ns[0]
                        break

        if not selected_ann:
            if not jam.annotations:
                raise ValueError(f"No annotations found in JAMS file: {file_path_for_error_msg}")
            available_ns = sorted(list(set(ann.namespace for ann in jam.annotations)))
            err_msg = (
                f"Cannot auto-load from {file_path_for_error_msg}. No default types "
                f"(e.g., multi_segment, {common_segment_namespaces}) found. "
                f"Available: {available_ns}. Specify 'annotation_id'."
            )
            raise ValueError(err_msg)

        if selected_ann.namespace == "multi_segment":
            return Hierarchy.from_jams(selected_ann)
        else:
            try:
                return Segmentation.from_jams(selected_ann)
            except Exception as e:
                raise ValueError(
                    f"Failed to load JAMS annotation (namespace: '{selected_ann.namespace}') as Segmentation "
                    f"from {file_path_for_error_msg}: {e}"
                ) from e

    def _load_json_annotation(self, file_content_stream: io.StringIO, file_path_for_error_msg: str | Path) -> Hierarchy:
        """Helper to load and process a JSON annotation."""
        import json  # Ensure json is imported

        try:
            json_data = json.load(file_content_stream)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path_for_error_msg}: {e}") from e
        return Hierarchy.from_json(json_data)


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
            self.base_url = str(manifest_path).rsplit("/", 1)[0]
            self.dataset_root = self.base_url
        else:
            raise ValueError(f"Unsupported data_location: '{self.data_location}'")

        try:
            # Use expanded path for local files, original path for URLs
            load_path = Path(manifest_path).expanduser() if self.data_location == "local" else manifest_path
            self.manifest = (
                pd.read_csv(io.StringIO(requests.get(str(manifest_path)).text))
                if self.data_location == "cloud"
                else pd.read_csv(load_path)
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}") from e

            # --- Standardize track_id to string ---
        if "track_id" not in self.manifest.columns:
            raise ValueError("Manifest must contain a 'track_id' column.")
        self.manifest["track_id"] = self.manifest["track_id"].astype(str)
        self.manifest.set_index("track_id", inplace=True, drop=False)

        # Sort by track_id, handling numeric vs. lexical gracefully
        track_ids_unique = self.manifest["track_id"].astype(str).unique()
        try:
            # Try numeric sort first
            self.track_ids = sorted(track_ids_unique, key=lambda x: int(x))
        except ValueError:
            # Fall back to lexical sort if any track_id is non-numeric
            self.track_ids = sorted(track_ids_unique)

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

    def _format_adobe_params(self, asset_subtype: str) -> str:
        """Convert adobe asset subtype to formatted parameters."""
        mu_gamma = asset_subtype.split("-")[1]
        # Convert format: mu1gamma1 -> mu_0.1_gamma_0.1
        if mu_gamma == "mu1gamma1":
            return "mu_0.1_gamma_0.1"
        elif mu_gamma == "mu5gamma5":
            return "mu_0.5_gamma_0.5"
        elif mu_gamma == "mu1gamma9":
            return "mu_0.1_gamma_0.9"
        else:
            return mu_gamma  # fallback

    def _reconstruct_local_path(self, track_id: str, asset_type: str, asset_subtype: str) -> Path:
        """Reconstruct local file path for an asset."""
        root = cast(Path, self.dataset_root)

        if asset_type == "audio":
            return root / "audio" / track_id / f"audio.{asset_subtype}"
        elif asset_type == "annotation":
            if asset_subtype == "reference":
                return root / "jams" / f"{track_id}.jams"
            elif "adobe" in asset_subtype:
                formatted_params = self._format_adobe_params(asset_subtype)
                subfolder = f"adobe/def_{formatted_params}"
                return root / subfolder / f"{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown local asset structure for: {asset_type}/{asset_subtype}")

    def _reconstruct_cloud_url(self, track_id: str, asset_type: str, asset_subtype: str) -> str:
        """Reconstruct cloud URL for an asset."""
        base = cast(str, self.base_url)

        if asset_type == "audio" and asset_subtype == "mp3":
            return f"{base}/slm-dataset/{track_id}/audio.mp3"
        elif asset_type == "annotation" and asset_subtype == "reference":
            return f"{base}/ref-jams/{track_id}.jams"
        elif asset_type == "annotation" and "adobe" in asset_subtype:
            formatted_params = self._format_adobe_params(asset_subtype)
            subfolder = f"adobe21-est/def_{formatted_params}"
            return f"{base}/{subfolder}/{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown cloud asset structure for: {asset_type}/{asset_subtype}")

    def _reconstruct_path(self, track_id: str, asset_type: str, asset_subtype: str) -> Path | str:
        """Reconstructs the full path or URL for an asset.

        This logic must align with the conventions used by the manifest builders
        (e.g., `scripts/build_local_manifest.py`).
        """
        if self.data_location == "local":
            return self._reconstruct_local_path(track_id, asset_type, asset_subtype)
        elif self.data_location == "cloud":
            return self._reconstruct_cloud_url(track_id, asset_type, asset_subtype)
        else:
            raise ValueError(f"Unknown asset structure for source type '{self.data_location}'")
