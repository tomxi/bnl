"""Generic data loading utilities and configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import jams


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and settings.

    Parameters
    ----------
    data_root : str or Path, optional
        Root directory containing all datasets. Defaults to ~/data.
    salami_annotations_dir : str or Path, optional
        Directory containing SALAMI JAMS files. Defaults to {data_root}/salami-jams.
    salami_audio_dir : str or Path, optional
        Directory containing SALAMI audio files. Defaults to {data_root}/salami/audio.
    adobe_estimations_dir : str or Path, optional
        Directory containing ADOBE estimations. Defaults to {data_root}/ISMIR21-Segmentations/SALAMI/.
    """

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


# Global configuration instance
_default_config = DatasetConfig()


def get_config() -> DatasetConfig:
    """Get the current dataset configuration."""
    return _default_config


def set_config(config: DatasetConfig) -> None:
    """Set the global dataset configuration."""
    global _default_config
    _default_config = config
