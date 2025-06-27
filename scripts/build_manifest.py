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
        asset_subtype = (
            spec["asset_subtype"] or file_path.suffix[1:].lower()
        )  # ensure lower for consistency e.g. .MP3 -> mp3

        records.append(
            {
                "track_id": track_id,
                "asset_type": spec["asset_type"],
                "asset_subtype": asset_subtype,
                # "file_path": str(file_path.relative_to(dataset_root)),
                # Not needed for boolean manifest
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

    # --- Convert to wide format (boolean flags) ---
    manifest_data = {}
    all_possible_asset_cols = set()

    for record in all_records:
        track_id = record["track_id"]
        if track_id not in manifest_data:
            manifest_data[track_id] = {"track_id": track_id}

        # Create a column name like has_<asset_type>_<asset_subtype>
        # e.g., has_audio_mp3 or has_annotation_reference
        asset_col_name = f"has_{record['asset_type']}_{record['asset_subtype']}"
        manifest_data[track_id][asset_col_name] = True
        all_possible_asset_cols.add(asset_col_name)

    # Ensure all tracks have all possible asset columns, defaulting to False
    final_rows = []
    sorted_track_ids = sorted(manifest_data.keys(), key=int)

    for track_id in sorted_track_ids:
        row = {"track_id": track_id}
        for col_name in sorted(list(all_possible_asset_cols)):  # Ensure consistent column order
            row[col_name] = manifest_data[track_id].get(col_name, False)
        final_rows.append(row)

    if not final_rows:
        print("Warning: No records to write to manifest. It will have headers only.")
        # Create a header-only CSV if no records
        fieldnames = ["track_id"] + sorted(list(all_possible_asset_cols))
    else:
        fieldnames = list(
            final_rows[0].keys()
        )  # Already includes track_id and sorted asset columns

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if final_rows:  # Only write rows if there are any
            writer.writerows(final_rows)

    print(f"\nManifest with {len(final_rows)} tracks written to: {output_path}")
    print(f"Columns in manifest: {fieldnames}")
    validate_manifest_boolean(final_rows, all_possible_asset_cols)


def validate_manifest_boolean(records: list[dict[str, Any]], asset_cols: set[str]):
    """Performs basic validation on the generated boolean manifest records."""
    if not records:
        return

    track_ids = {r["track_id"] for r in records}
    print(f"Unique tracks: {len(track_ids)}")

    # Example validation: Check if tracks with a specific annotation also have audio.
    # This depends on knowing specific column names, e.g., 'has_annotation_reference'
    # and 'has_audio_mp3'. Adjust based on ASSET_SPECS.

    # Infer primary audio and annotation columns if possible (this is a heuristic)
    primary_audio_col = next((col for col in asset_cols if col.startswith("has_audio_")), None)
    primary_annotation_col = next(
        (col for col in asset_cols if col.startswith("has_annotation_") and "reference" in col),
        None,
    )
    if not primary_annotation_col:  # Fallback if no 'reference'
        primary_annotation_col = next(
            (col for col in asset_cols if col.startswith("has_annotation_")), None
        )

    if primary_audio_col and primary_annotation_col:
        tracks_with_annotation = {
            r["track_id"] for r in records if r.get(primary_annotation_col)
        }
        tracks_with_audio = {r["track_id"] for r in records if r.get(primary_audio_col)}

        missing_audio = tracks_with_annotation - tracks_with_audio
        if missing_audio:
            print(
                f"Warning: {len(missing_audio)} tracks have '{primary_annotation_col}' "
                f"but no '{primary_audio_col}'."
            )

        missing_annotation = tracks_with_audio - tracks_with_annotation
        if missing_annotation:
            print(
                f"Warning: {len(missing_annotation)} tracks have '{primary_audio_col}' "
                f"but no '{primary_annotation_col}'."
            )
    else:
        print(
            "Skipping audio/annotation consistency check: Could not determine "
            "primary audio/annotation columns for validation."
        )


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
