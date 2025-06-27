"""Generic data loading utilities and configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jams
import librosa
import numpy as np
import pandas as pd


@dataclass
class BaseTrack:
    """A track with annotations and metadata.

    Parameters
    ----------
    track_id : str
        track identifier.
    audio_path : Path, optional
        Path to the audio file.
    jams_path : Path, optional
        Path to the JAMS file.
    """

    track_id: str
    audio_path: Optional[Path] = None
    jams_path: Optional[Path] = None

    def __post_init__(self):
        # Lazy loading - don't load JAMS until needed
        self._jams = None
        self._info = None

    @property
    def info(self) -> Dict[str, Any]:
        """Get track metadata from JAMS file."""
        if self._info is None:
            jams_obj = self.jams
            metadata = jams_obj.file_metadata
            self._info = {
                "artist": metadata.artist.replace("_", " ") if metadata.artist else "",
                "title": metadata.title.replace("_", " ") if metadata.title else "",
                "duration": metadata.duration,
            }
        return self._info

    @property
    def jams(self) -> jams.JAMS:
        """Load the JAMS file for the track."""
        if self._jams is None:
            if self.jams_path is None:
                raise ValueError("JAMS path is not set")
            self._jams = jams.load(str(self.jams_path))
        return self._jams

    def __repr__(self) -> str:
        try:
            info = self.info
            artist_title = ""
            if info.get("artist") and info.get("title"):
                artist_title = f" ({info['artist']} - {info['title']})"

            duration_str = f", {info['duration']:.1f}s" if info.get("duration") else ""
            return f"Track({self.track_id}{artist_title}{duration_str})"
        except Exception:
            # Fallback if JAMS loading fails
            return f"Track({self.track_id})"

    def load_audio(self, sr=22050) -> Tuple[np.ndarray, int]:
        """Load the audio file for the track."""
        if self.audio_path is None:
            raise ValueError("Audio path is not set")
        return librosa.load(self.audio_path, sr=sr, mono=True)


class BaseDataset:
    """Base class for manifest-based dataset loading.

    This class provides the core functionality for loading datasets
    from a metadata manifest file (CSV format).

    Parameters
    ----------
    manifest_path : str or Path
        Path to the metadata.csv manifest file
    """

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path)
        self._manifest_df = None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

    @property
    def manifest(self) -> pd.DataFrame:
        """Lazily load and cache the manifest DataFrame."""
        if self._manifest_df is None:
            # Fail fast: validate manifest format on first load
            try:
                self._manifest_df = pd.read_csv(self.manifest_path)
                required_columns = {
                    "track_id",
                    "asset_type",
                    "asset_subtype",
                    "file_path",
                }
                if not required_columns.issubset(self._manifest_df.columns):
                    missing = required_columns - set(self._manifest_df.columns)
                    raise ValueError(f"Manifest missing required columns: {missing}")
            except Exception as e:
                raise ValueError(f"Invalid manifest file {self.manifest_path}: {e}")

        return self._manifest_df

    def list_track_ids(self) -> List[str]:
        """Get all available track IDs from the manifest."""
        return sorted([str(tid) for tid in self.manifest["track_id"].unique()], key=int)

    def get_asset_path(
        self, track_id: str, asset_type: str, asset_subtype: str
    ) -> Optional[Path]:
        """Get the file path for a specific asset.

        Parameters
        ----------
        track_id : str
            Track identifier
        asset_type : str
            Type of asset (e.g., "audio", "annotation")
        asset_subtype : str
            Subtype of asset (e.g., "mix", "reference")

        Returns
        -------
        Path or None
            Path to the asset file, or None if not found
        """
        mask = (
            (self.manifest["track_id"].astype(str) == str(track_id))
            & (self.manifest["asset_type"] == asset_type)
            & (self.manifest["asset_subtype"] == asset_subtype)
        )

        matches = self.manifest[mask]
        if matches.empty:
            return None

        # Fail fast: should be exactly one match
        if len(matches) > 1:
            raise ValueError(
                f"Multiple assets found for {track_id}/{asset_type}/{asset_subtype}"
            )

        file_path = Path(matches.iloc[0]["file_path"])

        # Fail fast: referenced file must exist
        if not file_path.exists():
            raise FileNotFoundError(f"Asset file not found: {file_path}")

        return file_path

    def get_audio_path(self, track_id: str) -> Optional[Path]:
        """Get the audio file path for a track."""
        return self.get_asset_path(track_id, "audio", "mix")

    def get_jams_path(self, track_id: str) -> Optional[Path]:
        """Get the JAMS annotation file path for a track."""
        return self.get_asset_path(track_id, "annotation", "reference")


# Legacy configuration support - will be deprecated
@dataclass
class DatasetConfig:
    """Legacy configuration for dataset paths and settings."""

    data_root: Optional[Path] = None
    salami_annotations_dir: Optional[Path] = None
    salami_audio_dir: Optional[Path] = None
    adobe_estimations_dir: Optional[Path] = None

    def __post_init__(self):
        # Set default data root
        if self.data_root is None:
            self.data_root = Path.home() / "data"
        else:
            self.data_root = Path(self.data_root)

        # Set default paths if not provided
        if self.salami_annotations_dir is None:
            self.salami_annotations_dir = self.data_root / "salami-jams"
        else:
            self.salami_annotations_dir = Path(self.salami_annotations_dir)

        if self.salami_audio_dir is None:
            self.salami_audio_dir = self.data_root / "salami" / "audio"
        else:
            self.salami_audio_dir = Path(self.salami_audio_dir)

        if self.adobe_estimations_dir is None:
            self.adobe_estimations_dir = self.data_root / "adobe"
        else:
            self.adobe_estimations_dir = Path(self.adobe_estimations_dir)


# Global configuration instance (legacy)
_default_config = DatasetConfig()


def build_config(data_root: Path) -> DatasetConfig:
    """Build a dataset configuration from a data root."""
    return DatasetConfig(data_root=data_root)


def get_config() -> DatasetConfig:
    """Get the current dataset configuration."""
    return _default_config


def set_config(config: DatasetConfig) -> None:
    """Set the global dataset configuration."""
    global _default_config
    _default_config = config
