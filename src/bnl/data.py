"""Core data loading classes for manifest-based datasets."""

from __future__ import annotations

__all__ = [
    "Track",
    "Dataset",
]

import io
import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

import jams
import numpy as np
import pandas as pd
import requests

from .core import MultiSegment, Segment
from .lsd import run as run_lsd
from .relevance import cd_suite, rel_suite


@dataclass
class Track:
    """A single track and its associated data assets."""

    track_id: str
    manifest_row: pd.Series
    dataset: Dataset

    def __repr__(self) -> str:
        has_columns = self.manifest_row.filter(like="has_").astype(bool)
        num_assets = int(has_columns.values.sum()) if len(has_columns) > 0 else 0
        return (
            f"Track(track_id='{self.track_id}', num_assets={num_assets}, "
            f"source='{self.dataset.data_location}')"
        )

    @cached_property
    def info(self) -> dict[str, Any]:
        """Essential track information (cached)."""

        info: dict[str, Any] = {"track_id": self.track_id}

        # Reconstruct paths for all available assets
        for col_name, has_asset in self.manifest_row.items():
            if str(col_name).startswith("has_") and has_asset:
                parts = col_name.replace("has_", "").split("_", 1)
                asset_type = parts[0]
                asset_subtype = parts[1] if len(parts) > 1 else None

                if asset_type and asset_subtype:
                    path_or_url = self.dataset._reconstruct_path(
                        self.track_id, asset_type, asset_subtype
                    )
                    info[f"{asset_type}_{asset_subtype}_path"] = path_or_url

        return info

    @cached_property
    def refs(self) -> dict[str, MultiSegment]:
        """Returns available reference annotations."""
        # Get the jams reference file and find all the annotators
        # Add JAMS metadata if reference annotation exists
        if self.jam is not None:
            annotators = [
                ann.annotation_metadata.annotator.name
                for ann in self.jam.search(namespace="segment_salami_function")
            ]
            annotators = list(set(annotators))
        else:
            annotators = []
        return {a_id: self.load_annotation("reference", a_id) for a_id in annotators}

    @property
    def ref(self) -> MultiSegment:
        """Returns the first reference annotation."""
        return self.refs[list(self.refs)[0]]

    @cached_property
    def ests(self) -> dict[str, MultiSegment]:
        """Returns available estimated annotations."""

        # Find all available estimated annotations from info
        est_keys = [key for key in self.info if key.startswith("annotation_adobe")]
        est_ids = [key.replace("annotation_adobe-", "").replace("_path", "") for key in est_keys]

        return {est_id: self.load_annotation(f"adobe-{est_id}") for est_id in est_ids}

    @cached_property
    def jam(self) -> jams.JAMS | None:
        """Returns the reference JAMS object for this track."""
        jam_path = self.info.get("annotation_reference_path")
        if jam_path is not None:
            return jams.load(self._fetch_content(jam_path))
        return None

    @property
    def feats(self) -> np.NpzFile:
        audio_path = str(self.info["audio_mp3_path"])
        feat_path = audio_path.replace("/audio.mp3", "_synced_feats.npz").replace("audio", "feats")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"{feat_path} does not exist...")
        else:
            return np.load(feat_path)

    @property
    def feat_types(self) -> list[str]:
        return ["tempogram", "crema", "yamnet", "openl3", "mfcc"]

    def lsds(
        self,
        rep_feats: tuple[str] | None = None,
        loc_feats: tuple[str] | None = None,
    ):
        # check if lsds already exists
        lsds_path = (
            Path(self.dataset.manifest_path).expanduser().parent / "lsds" / f"{self.track_id}.json"
        )
        if lsds_path.exists():
            # load json file as dict
            with open(lsds_path) as f:
                all_outs = json.load(f)
        else:
            os.makedirs(os.path.dirname(lsds_path), exist_ok=True)
            # run lsd for all combinations of features
            all_outs = dict()
            for rep_feat in self.feat_types:
                for loc_feat in self.feat_types:
                    print(rep_feat, loc_feat)
                    all_outs[f"{rep_feat}_{loc_feat}"] = run_lsd(
                        self.feats,
                        rep_feat=rep_feat,
                        loc_feat=loc_feat,
                    ).to_json()

            # save json file
            with open(lsds_path, "w") as f:
                json.dump(all_outs, f)

        # convert each value to MultiSegment before returning
        if rep_feats is None:
            rep_feats = self.feat_types
        if loc_feats is None:
            loc_feats = self.feat_types
        return {
            k: MultiSegment.from_json(v, name=self.track_id + "-" + k)
            for k, v in all_outs.items()
            if k.split("_")[0] in rep_feats and k.split("_")[1] in loc_feats
        }

    def lsd_relevance(self):
        import pickle

        # check if relevance already calculated
        rel_path = (
            Path(self.dataset.manifest_path).expanduser().parent
            / "lsd_rel"
            / f"{self.track_id}.pkl"
        )
        if rel_path.exists():
            # load pickle file as dict
            with open(rel_path, "rb") as f:
                return pickle.load(f)
        else:
            out = dict()
            for ref_name, ref in self.refs.items():
                out[ref_name] = rel_suite(ref, self.lsds())
            # save pickle file
            os.makedirs(os.path.dirname(rel_path), exist_ok=True)
            with open(rel_path, "wb") as f:
                pickle.dump(out, f)
            return out

    def lsd_cds(self, recompute=False):
        import pickle

        # check if lsd_cd already calculated
        cd_path = (
            Path(self.dataset.manifest_path).expanduser().parent
            / "lsd_cd2"
            / f"{self.track_id}.pkl"
        )
        if not recompute and cd_path.exists():
            # load pickle file as dict
            with open(cd_path, "rb") as f:
                return pickle.load(f)
        else:
            cds = cd_suite(self.lsds())
            os.makedirs(os.path.dirname(cd_path), exist_ok=True)
            with open(cd_path, "wb") as f:
                pickle.dump(cds, f)
            return cds

    def load_annotation(self, annotation_type: str, annotator: str | None = None) -> MultiSegment:
        """Loads a specific annotation as a MultiSegment.

        Parameters:
            annotation_type (str): One of:
                - 'reference' to load the reference JAMS. Optionally pass
                  `annotator` to select a specific annotator by name.
                - 'adobe-<id>' to load an Adobe JSON (e.g., 'adobe-mu1gamma1').
            annotator (str | None): Name of the annotator in the JAMS file to select.

        Raises:
            ValueError: If the requested annotation is unavailable for this track.
            NotImplementedError: If the file type is unsupported (.jams and .json are supported).

        """
        annotation_key = f"annotation_{annotation_type}_path"
        if annotation_key not in self.info:
            raise ValueError(f"Annotation type '{annotation_type}' not available for this track.")

        annotation_path = self.info[annotation_key]

        if str(annotation_path).lower().endswith(".jams"):
            return self._load_jams_anno(annotation_path, name=annotator)
        elif str(annotation_path).lower().endswith(".json"):
            return self._load_json(annotation_path, name=f"{self.track_id}-{annotation_type}")
        else:
            raise NotImplementedError(f"Unsupported file type: {annotation_path}")

    def _load_jams_anno(self, path: str | Path, name: str | None = None) -> MultiSegment:
        """
        Find the annotator with name `name` in the JAMS file, and load it as a `MultiSegment`.
        If `name` is None, find the first annotator in the JAMS file.
        Each MultiSegment contains two layers:
        - coarse (`segment_salami_function`)
        - fine (`segment_salami_lower`)
        """
        jam = jams.load(self._fetch_content(path))
        search_name = name if name is not None else ""
        uppers = jam.search(namespace="segment_salami_function").search(name=search_name)
        lowers = jam.search(namespace="segment_salami_lower").search(name=search_name)

        if len(uppers) == 0 or len(lowers) == 0:
            raise ValueError(f"No annotator found for {name}")

        return MultiSegment(
            raw_layers=[
                Segment.from_jams(uppers[0], name="coarse"),
                Segment.from_jams(lowers[0], name="fine"),
            ],
            name=f"{self.track_id}-annotator-{uppers[0].annotation_metadata.annotator.name}",
        )

    def _load_json(self, path: str | Path, name: str | None = None) -> MultiSegment:
        """Loads a JSON annotation as a MultiSegment."""
        json_data = json.load(self._fetch_content(path))
        ms_name = f"{self.track_id} JSON Annotation" if name is None else name
        return MultiSegment.from_json(json_data, name=ms_name)

    @staticmethod
    def _fetch_content(path: str | Path) -> io.StringIO:
        """Fetches file content into a memory buffer. works for local files and urls."""
        if isinstance(path, str) and path.startswith("http"):
            response = requests.get(
                str(path),
                timeout=float(os.getenv("BNL_HTTP_TIMEOUT", "10")),
                headers={"User-Agent": "bnl"},
            )
            response.raise_for_status()
            return io.StringIO(response.text)
        elif Path(path).exists():
            with open(path, encoding="utf-8") as f:
                return io.StringIO(f.read())
        else:
            raise FileNotFoundError(f"File not found: {path}")


@dataclass
class SpamTrack:
    """A track in the SPAM dataset."""

    track_id: str
    manifest_row: pd.Series
    dataset: Dataset

    @cached_property
    def refs(self) -> dict[str, MultiSegment]:
        refs = {}
        for name in SpamDataset.ANNOTATOR_NAMES:
            upper_annos = self.jam.search(namespace="segment_salami_upper", name=name)
            lower_annos = self.jam.search(namespace="segment_salami_lower", name=name)

            first_name = name.split(" ")[0]
            refs[first_name] = MultiSegment(
                raw_layers=[
                    Segment.from_jams(upper_annos[0], name="coarse"),
                    Segment.from_jams(lower_annos[0], name="fine"),
                ],
                name=f"{self.track_id}-{first_name}",
            )
        return refs

    @property
    def info(self):
        return self.manifest_row

    @property
    def ref(self) -> MultiSegment:
        """Returns the first reference annotation."""
        return self.refs["Colin"]

    @cached_property
    def ests(self) -> dict[str, MultiSegment]:
        pass

    @cached_property
    def jam(self) -> jams.JAMS | None:
        # construct jams path from where the manifest was
        jams_path = (
            self.dataset.manifest_path.parent
            / "references"
            / self.manifest_row["File Name"].replace(".mp3", ".jams")
        )
        if jams_path.exists():
            return jams.load(str(jams_path), validate=False)
        else:
            raise FileNotFoundError(f"JAMS file not found: {jams_path}")

    @property
    def feats(self) -> pd.DataFrame:
        feat_path = (
            self.dataset.manifest_path.parent
            / "features"
            / self.manifest_row["File Name"].replace(".mp3", ".json")
        )
        return feat_path


class Dataset:
    """A manifest-based dataset."""

    track_ids: list[str]
    manifest: pd.DataFrame
    manifest_path: Path

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

    def lucky(self) -> Track:
        """Return a random track."""
        return self[random.choice(self.track_ids)]


class SalamiDataset(Dataset):
    """A manifest-based dataset."""

    # Allow overriding the public bucket via environment for easy configuration in
    # local development without code changes.
    R2_BUCKET_PUBLIC_URL: str = os.getenv(
        "BNL_R2_BUCKET_PUBLIC_URL",
        "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev",
    )

    data_location: Literal["local", "cloud"] = field(init=False)

    def __init__(self, manifest_path: Path | str | None = None):
        if manifest_path is None:
            manifest_path = f"{self.R2_BUCKET_PUBLIC_URL}/manifest_cloud_boolean.csv"
        self.manifest_path = manifest_path
        self.data_location = (
            "cloud"
            if isinstance(manifest_path, str) and manifest_path.startswith("http")
            else "local"
        )

        if self.data_location == "local":
            expanded_path = Path(manifest_path).expanduser()
            self.dataset_root: Path | str = expanded_path.parent
            self.base_url: str | None = None
        else:
            self.base_url = str(manifest_path).rsplit("/", 1)[0]
            self.dataset_root = self.base_url

        # Load manifest
        try:
            load_path = (
                Path(manifest_path).expanduser() if self.data_location == "local" else manifest_path
            )
            if self.data_location == "cloud":
                # Minimal robustness: use a short timeout and a simple User-Agent.
                response = requests.get(
                    str(manifest_path),
                    timeout=float(os.getenv("BNL_HTTP_TIMEOUT", "10")),
                    headers={"User-Agent": "bnl"},
                )
                response.raise_for_status()
                self.manifest = pd.read_csv(io.StringIO(response.text))
            else:
                self.manifest = pd.read_csv(load_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}") from e

        self.manifest["track_id"] = self.manifest["track_id"].astype(str)
        self.manifest.set_index("track_id", inplace=True, drop=False)

        # Only include tracks that have the reference annotation
        self.manifest = self.manifest[
            self.manifest.filter(like="has_annotation_reference").astype(bool).values.any(axis=1)
        ]

        try:
            self.track_ids = sorted(self.manifest["track_id"].unique(), key=int)
        except ValueError:
            self.track_ids = sorted(self.manifest["track_id"].unique())

    @staticmethod
    def _format_adobe_params(asset_subtype: str) -> str:
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
            return root / "audio" / track_id / f"audio.{asset_subtype}"
        elif asset_type == "annotation":
            if asset_subtype.startswith("ref_") or asset_subtype == "reference":
                return root / "jams" / f"{track_id}.jams"
            elif "adobe" in asset_subtype:
                # Adobe annotations have a specific subfolder structure.
                subfolder = f"adobe/def_{self._format_adobe_params(asset_subtype)}"
                return root / subfolder / f"{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown local asset: {asset_type}/{asset_subtype}")

    def _reconstruct_cloud_url(self, track_id: str, asset_type: str, asset_subtype: str) -> str:
        """Reconstruct cloud URL for an asset."""
        base = cast(str, self.base_url)

        if asset_type == "audio":
            return f"{base}/slm-dataset/{track_id}/audio.{asset_subtype}"
        elif asset_type == "annotation" and (
            asset_subtype.startswith("ref_") or asset_subtype == "reference"
        ):
            return f"{base}/ref-jams/{track_id}.jams"
        elif asset_type == "annotation" and "adobe" in asset_subtype:
            subfolder = f"adobe21-est/def_{self._format_adobe_params(asset_subtype)}"
            return f"{base}/{subfolder}/{track_id}.mp3.msdclasscsnmagic.json"

        raise ValueError(f"Unknown cloud asset: {asset_type}/{asset_subtype}")

    def _reconstruct_path(self, track_id: str, asset_type: str, asset_subtype: str) -> Path | str:
        """
        Reconstruct the full path or URL for an asset based on the dataset's
        data_location.

        Dispatches to _reconstruct_local_path for local assets or
        _reconstruct_cloud_url for cloud assets.
        """
        if self.data_location == "local":
            return self._reconstruct_local_path(track_id, asset_type, asset_subtype)
        elif self.data_location == "cloud":
            return self._reconstruct_cloud_url(track_id, asset_type, asset_subtype)
        else:
            raise ValueError(f"Unknown data location: {self.data_location}")


class SpamDataset(Dataset):
    """A manifest-based dataset."""

    ANNOTATOR_NAMES = [
        "Colin Hua",
        "Eleni Vasilia Maltas",
        "Evan S. Johnson",
        "John Turner",
        "Shuli Tang",
    ]

    def __init__(self, manifest_path: Path | str = "~/code/msaf-data/SPAM/metadata.tsv") -> None:
        try:
            self.manifest_path = Path(manifest_path).expanduser()
            self.manifest = pd.read_csv(self.manifest_path, delimiter="\t")
            self.manifest.rename(columns={"id": "track_id"}, inplace=True)
            self.manifest["track_id"] = self.manifest["track_id"].astype(str)
            self.manifest.set_index("track_id", inplace=True, drop=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}") from e

        try:
            self.track_ids = sorted(self.manifest.index.unique(), key=int)
        except ValueError:
            self.track_ids = sorted(self.manifest.index.unique())

    def __getitem__(self, track_id: str) -> Track:
        """Load a specific track by its ID."""
        track_id = str(track_id)
        if track_id not in self.track_ids:
            raise ValueError(f"Track ID '{track_id}' not found in manifest.")
        return SpamTrack(track_id, self.manifest.loc[track_id], self)
