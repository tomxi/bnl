"""Core data loading classes for manifest-based datasets."""

from __future__ import annotations

import io
import json
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

from .core import MultiSegment, Segment


def _parse_jams_metadata(jams_path: Path | str) -> dict[str, Any]:
    """Loads metadata from a JAMS file."""
    try:
        if isinstance(jams_path, str) and jams_path.startswith("http"):
            headers = {"User-Agent": "BNL-Dataset/1.0 (Music Information Retrieval)"}
            response = requests.get(jams_path, headers=headers)
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
    """A single track and its associated data assets."""

    track_id: str
    manifest_row: pd.Series
    dataset: Dataset

    def __post_init__(self) -> None:
        self._info_cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        has_columns = self.manifest_row.filter(like="has_").astype(bool)
        num_assets = int(has_columns.values.sum()) if len(has_columns) > 0 else 0
        return f"Track(track_id='{self.track_id}', num_assets={num_assets}, source='{self.dataset.data_location}')"

    @property
    def info(self) -> dict[str, Any]:
        """Essential track information (cached)."""
        if self._info_cache is not None:
            return self._info_cache

        info: dict[str, Any] = {"track_id": self.track_id}

        # Reconstruct paths for all available assets
        for col_name, has_asset in self.manifest_row.items():
            if str(col_name).startswith("has_") and has_asset:
                parts = col_name.replace("has_", "").split("_", 1)
                asset_type = parts[0]
                asset_subtype = parts[1] if len(parts) > 1 else None

                if asset_type and asset_subtype:
                    path_or_url = self.dataset._reconstruct_path(self.track_id, asset_type, asset_subtype)
                    info[f"{asset_type}_{asset_subtype}_path"] = path_or_url

        # Add JAMS metadata if reference annotation exists
        if jams_path_or_url := info.get("annotation_reference_path"):
            info.update(_parse_jams_metadata(jams_path_or_url))

        self._info_cache = info
        return self._info_cache

    @property
    def has_annotations(self) -> bool:
        """Checks if the track has any annotations."""
        return self.manifest_row.filter(like="has_annotation_").any()

    @property
    def annotations(self) -> dict[str, str | Path]:
        """Returns available annotation paths."""
        ann_paths = {}
        for key, value in self.info.items():
            if key.startswith("annotation_") and key.endswith("_path"):
                ann_type = key.replace("annotation_", "").replace("_path", "")
                ann_paths[ann_type] = value
        return ann_paths

    def load_audio(self) -> tuple[np.ndarray | None, float | None]:
        """Loads the track's audio waveform and sample rate."""
        # Find the first audio asset
        audio_path_key = next(
            (key for key in self.info.keys() if key.startswith("audio_") and key.endswith("_path")), None
        )

        if not audio_path_key:
            return None, None

        audio_path = self.info[audio_path_key]
        expanded_path = Path(audio_path).expanduser() if isinstance(audio_path, str | Path) else audio_path

        try:
            if isinstance(audio_path, str) and audio_path.startswith("http"):
                headers = {"User-Agent": "BNL-Dataset/1.0 (Music Information Retrieval)"}
                response = requests.get(audio_path, headers=headers)
                response.raise_for_status()
                y, sr = librosa.load(io.BytesIO(response.content), sr=None, mono=True)
            elif expanded_path.exists():
                y, sr = librosa.load(expanded_path, sr=None, mono=True)
            else:
                print(f"Warning: Audio file not found at: {audio_path}")
                return None, None
            return y, sr
        except Exception as e:
            print(f"Warning: Failed to load audio from {audio_path}: {e}")
            return None, None

    def load_annotation(self, annotation_type: str, annotation_id: str | int | None = None) -> MultiSegment:
        """Loads a specific annotation as a `MultiSegment`."""
        if annotation_type not in self.annotations:
            raise ValueError(
                f"Annotation type '{annotation_type}' not available. Available: {list(self.annotations.keys())}"
            )

        annotation_path = self.annotations[annotation_type]

        try:
            content = self._fetch_content(annotation_path)
        except Exception as e:
            raise ValueError(f"Failed to fetch annotation from {annotation_path}: {e}") from e

        if str(annotation_path).lower().endswith(".jams"):
            return self._load_jams(content, Path(annotation_path), annotation_id)
        elif str(annotation_path).lower().endswith(".json"):
            return self._load_json(content, name=annotation_type)
        else:
            raise NotImplementedError(f"Unsupported file type: {annotation_path}")

    def _fetch_content(self, path: str | Path) -> io.StringIO:
        """Fetches file content into a memory buffer."""
        if isinstance(path, str) and path.startswith("http"):
            headers = {"User-Agent": "BNL-Dataset/1.0 (Music Information Retrieval)"}
            response = requests.get(str(path), headers=headers)
            response.raise_for_status()
            return io.StringIO(response.text)
        elif Path(path).exists():
            with open(path, encoding="utf-8") as f:
                return io.StringIO(f.read())
        else:
            raise FileNotFoundError(f"File not found: {path}")

    def _load_jams(self, content: io.StringIO, path: Path, annotation_id: str | int | None) -> MultiSegment:
        """
        Loads a JAMS annotation as a two-layer MultiSegment containing coarse
        (`segment_salami_function`) and fine (`segment_salami_lower`) layers.
        """
        jam = jams.load(content)
        path_str = str(path)

        coarse_ann = self._find_salami_annotation(jam, "segment_salami_function", annotation_id, path_str)
        fine_ann = self._find_salami_annotation(jam, "segment_salami_lower", annotation_id, path_str)

        coarse_seg = Segment.from_jams(coarse_ann, name="coarse")
        fine_seg = Segment.from_jams(fine_ann, name="fine")

        return MultiSegment(
            layers=MultiSegment.align_layers([coarse_seg, fine_seg]),
            name=f"ref-{coarse_ann.annotation_metadata.annotator.name}",
        )

    def _find_salami_annotation(
        self, jam: jams.JAMS, namespace: str, annotator_id: str | int | None, path: str
    ) -> jams.Annotation:
        """Finds a Salami annotation, optionally filtering by annotator."""
        candidates = [ann for ann in jam.annotations if ann.namespace == namespace]

        if not candidates:
            raise ValueError(f"No '{namespace}' annotation found in {path}")

        if annotator_id is None:
            return candidates[0]

        if isinstance(annotator_id, int):
            if 0 <= annotator_id < len(candidates):
                return candidates[annotator_id]
            raise ValueError(f"Annotator index {annotator_id} out of range for '{namespace}' in {path}")

        # If annotator_id is a string, search in metadata
        for ann in candidates:
            annotator_meta = ann.annotation_metadata.annotator
            if isinstance(annotator_meta, dict) and annotator_meta.get("name") == annotator_id:
                return ann

        raise ValueError(f"No annotator '{annotator_id}' found for '{namespace}' in {path}")

    def _load_json(self, content: io.StringIO, name: str) -> MultiSegment:
        """Loads a JSON annotation as a MultiSegment."""
        try:
            json_data = json.load(content)
            return MultiSegment.from_json(json_data, name=name)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e


class Dataset:
    """A manifest-based dataset."""

    def __init__(self, manifest_path: Path | str):
        self.manifest_path = manifest_path
        self.data_location = "cloud" if urlparse(str(manifest_path)).scheme in ("http", "https") else "local"

        if self.data_location == "local":
            expanded_path = Path(manifest_path).expanduser()
            self.dataset_root: Path | str = expanded_path.parent
            self.base_url: str | None = None
        else:
            self.base_url = str(manifest_path).rsplit("/", 1)[0]
            self.dataset_root = self.base_url

        # Load manifest
        try:
            load_path = Path(manifest_path).expanduser() if self.data_location == "local" else manifest_path
            if self.data_location == "cloud":
                headers = {"User-Agent": "BNL-Dataset/1.0 (Music Information Retrieval)"}
                response = requests.get(str(manifest_path), headers=headers)
                response.raise_for_status()
                self.manifest = pd.read_csv(io.StringIO(response.text))
            else:
                self.manifest = pd.read_csv(load_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}") from e

        # Validate and process manifest
        if "track_id" not in self.manifest.columns:
            raise ValueError("Manifest must contain a 'track_id' column.")

        self.manifest["track_id"] = self.manifest["track_id"].astype(str)
        self.manifest.set_index("track_id", inplace=True, drop=False)

        # Sort track IDs
        try:
            self.track_ids = sorted(self.manifest["track_id"].unique(), key=int)
        except ValueError:
            self.track_ids = sorted(self.manifest["track_id"].unique())

    def __getitem__(self, track_id: str) -> Track:
        """Load a specific track by its ID."""
        track_id = str(track_id)
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
        if mu_gamma == "mu1gamma1":
            return "mu_0.1_gamma_0.1"
        elif mu_gamma == "mu5gamma5":
            return "mu_0.5_gamma_0.5"
        elif mu_gamma == "mu1gamma9":
            return "mu_0.1_gamma_0.9"
        else:
            return mu_gamma

    def _reconstruct_local_path(self, track_id: str, asset_type: str, asset_subtype: str) -> Path:
        """Reconstruct local file path for an asset."""
        root = cast(Path, self.dataset_root)

        if asset_type == "audio":
            # This path is a placeholder as audio is not the focus of current tests.
            return root / "audio" / f"{track_id}.{asset_subtype}"
        elif asset_type == "annotation":
            if asset_subtype.startswith("ref_") or asset_subtype == "reference":
                # Reference JAMS annotations are located in the jams/ directory for local datasets
                return root / "jams" / f"{track_id}.jams"
            elif "adobe" in asset_subtype:
                # Adobe annotations have a specific subfolder structure.
                formatted_params = self._format_adobe_params(asset_subtype)
                subfolder = f"adobe/def_{formatted_params}"
                return root / subfolder / f"{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown local asset: {asset_type}/{asset_subtype}")

    def _reconstruct_cloud_url(self, track_id: str, asset_type: str, asset_subtype: str) -> str:
        """Reconstruct cloud URL for an asset."""
        base = cast(str, self.base_url)

        if asset_type == "audio" and asset_subtype == "mp3":
            return f"{base}/slm-dataset/{track_id}/audio.mp3"
        elif asset_type == "annotation" and (asset_subtype.startswith("ref_") or asset_subtype == "reference"):
            return f"{base}/ref-jams/{track_id}.jams"
        elif asset_type == "annotation" and "adobe" in asset_subtype:
            formatted_params = self._format_adobe_params(asset_subtype)
            subfolder = f"adobe21-est/def_{formatted_params}"
            return f"{base}/{subfolder}/{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown cloud asset: {asset_type}/{asset_subtype}")

    def _reconstruct_path(self, track_id: str, asset_type: str, asset_subtype: str) -> Path | str:
        """Reconstruct the full path or URL for an asset."""
        if self.data_location == "local":
            return self._reconstruct_local_path(track_id, asset_type, asset_subtype)
        elif self.data_location == "cloud":
            return self._reconstruct_cloud_url(track_id, asset_type, asset_subtype)
        else:
            raise ValueError(f"Unknown data location: {self.data_location}")
