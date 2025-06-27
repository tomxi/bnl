#!/usr/bin/env python3
"""Generic metadata manifest builder for datasets.

This script scans a dataset directory and creates a metadata.csv file
based on a flexible configuration of asset types. It's designed to be
easily adaptable to various directory structures and data types.

Usage:
    python scripts/build_manifest.py <dataset_root>
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

# --- Asset Specifications ---
# This list defines what assets to look for. Each dictionary specifies
# an asset type, where to find it, and how to extract its track ID.
# This is the primary place to add new data assets (e.g., new annotations).

ASSET_SPECS: list[dict[str, Any]] = [
    {
        "asset_type": "audio",
        "asset_subtype": None,  # Will be inferred from file extension
        "path_glob": "audio/*/*",  # Matches any file in subdirs of 'audio'
        "track_id_pattern": r"audio/(?P<track_id>\d+)/",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "reference",
        "path_glob": "jams/*.jams",
        "track_id_pattern": r"jams/(?P<track_id>\d+)\.jams",
    },
]


def find_assets(dataset_root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Find all assets matching a specification.

    Args:
        dataset_root: The root directory of the dataset.
        spec: An asset specification dictionary.

    Returns:
        A list of record dictionaries for the found assets.
    """
    records = []
    asset_glob = spec["path_glob"]
    track_id_pattern = re.compile(spec["track_id_pattern"])

    for file_path in dataset_root.glob(asset_glob):
        if not file_path.is_file():
            continue

        # Use forward slashes for cross-platform regex matching
        path_str = str(file_path.relative_to(dataset_root).as_posix())
        match = track_id_pattern.search(path_str)

        if not match:
            # print(f"Warning: No track_id match for {path_str}")
            continue

        track_id = match.group("track_id")
        asset_subtype = spec["asset_subtype"] or file_path.suffix[1:]

        records.append(
            {
                "track_id": track_id,
                "asset_type": spec["asset_type"],
                "asset_subtype": asset_subtype,
                "file_path": str(file_path.relative_to(dataset_root)),
            }
        )
    return records


def build_manifest(dataset_root: Path, output_path: Path) -> None:
    """Build metadata manifest for a dataset based on ASSET_SPECS.

    Parameters:
        dataset_root: Path to the dataset root directory.
        output_path: Path where to write the metadata.csv file.
    """
    print(f"Scanning dataset at: {dataset_root}")
    all_records = []
    for spec in ASSET_SPECS:
        print(
            f"  - Finding '{spec['asset_type']}/{spec.get('asset_subtype', '*')}' "
            f"assets (glob: '{spec['path_glob']}')"
        )
        assets = find_assets(dataset_root, spec)
        all_records.extend(assets)
        print(f"    Found {len(assets)} assets.")

    if not all_records:
        print("Warning: No assets found. Manifest will be empty.")
        return

    # Sort by track_id and then by asset type for consistent ordering
    all_records.sort(key=lambda r: (int(r["track_id"]), r["asset_type"]))

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        fieldnames = ["track_id", "asset_type", "asset_subtype", "file_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nManifest with {len(all_records)} records written to: {output_path}")
    validate_manifest(all_records)


def validate_manifest(records: list[dict[str, Any]]):
    """Performs basic validation on the generated manifest records."""
    if not records:
        return

    track_ids = {r["track_id"] for r in records}
    print(f"Unique tracks: {len(track_ids)}")

    assets_by_type = {}
    for r in records:
        assets_by_type.setdefault(r["asset_type"], set()).add(r["track_id"])

    # Basic check for mismatched assets between primary types
    if "audio" in assets_by_type and "annotation" in assets_by_type:
        audio_tracks = assets_by_type["audio"]
        anno_tracks = assets_by_type["annotation"]

        missing_audio = anno_tracks - audio_tracks
        if missing_audio:
            print(f"Warning: {len(missing_audio)} tracks have annotations but no audio.")

        missing_jams = audio_tracks - anno_tracks
        if missing_jams:
            print(f"Warning: {len(missing_jams)} tracks have audio but no annotations.")


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset manifest.")
    parser.add_argument("dataset_root", type=Path, help="Path to the dataset root directory.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for metadata.csv (default: {dataset_root}/metadata.csv)",
    )

    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        print(f"Error: Dataset root not found or not a directory: {args.dataset_root}")
        return 1

    output_path = args.output or args.dataset_root / "metadata.csv"

    try:
        build_manifest(args.dataset_root, output_path)
        return 0
    except Exception as e:
        print(f"Error building manifest: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
