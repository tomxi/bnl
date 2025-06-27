#!/usr/bin/env python3
"""A dedicated script to build a manifest for the local SALAMI dataset."""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

# --- Asset Specifications for Local SALAMI Dataset ---
ASSET_SPECS: list[dict[str, Any]] = [
    {
        "asset_type": "audio",
        "asset_subtype": None,  # Inferred from file extension
        "path_glob": "audio/*/*",  # e.g., audio/1/audio.mp3
        "track_id_pattern": r"audio/(?P<track_id>\d+)/",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "reference",
        "path_glob": "jams/*.jams",  # e.g., jams/1.jams
        "track_id_pattern": r"jams/(?P<track_id>\d+)\.jams",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "adobe-mu1gamma1",
        "path_glob": "adobe/def_mu_0.1_gamma_0.1/*.json",
        "track_id_pattern": r"adobe/def_mu_0\.1_gamma_0\.1/(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "adobe-mu5gamma5",
        "path_glob": "adobe/def_mu_0.5_gamma_0.5/*.json",
        "track_id_pattern": r"adobe/def_mu_0\.5_gamma_0\.5/(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "adobe-mu1gamma9",
        "path_glob": "adobe/def_mu_0.1_gamma_0.9/*.json",
        "track_id_pattern": r"adobe/def_mu_0\.1_gamma_0\.9/(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json",
    },
]


def find_assets(dataset_root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Find all assets matching a specification."""
    records = []
    asset_glob = spec["path_glob"]
    track_id_pattern = re.compile(spec["track_id_pattern"])

    for file_path in dataset_root.glob(asset_glob):
        if not file_path.is_file():
            continue

        # Use the relative path for pattern matching
        path_str = str(file_path.relative_to(dataset_root).as_posix())

        match = track_id_pattern.search(path_str)

        if not match:
            continue

        track_id = match.group("track_id")
        asset_subtype = spec["asset_subtype"] or file_path.suffix[1:].lower()

        records.append(
            {
                "track_id": track_id,
                "asset_type": spec["asset_type"],
                "asset_subtype": asset_subtype,
            }
        )
    return records


def build_manifest(dataset_root: Path, output_path: Path) -> None:
    """Build metadata manifest for the local dataset."""
    print(f"Scanning dataset at: {dataset_root}")
    all_records = []
    for spec in ASSET_SPECS:
        print(
            f"  - Finding '{spec['asset_type']}/{spec.get('asset_subtype', '*')}' assets (glob: '{spec['path_glob']}')"
        )
        assets = find_assets(dataset_root, spec)
        all_records.extend(assets)
        print(f"    Found {len(assets)} assets.")

    if not all_records:
        print("Warning: No assets found. Manifest will be empty.")
        return

    manifest_data = {}
    all_possible_asset_cols = set()
    for record in all_records:
        track_id = record["track_id"]
        if track_id not in manifest_data:
            manifest_data[track_id] = {"track_id": track_id}
        asset_col_name = f"has_{record['asset_type']}_{record['asset_subtype']}"
        manifest_data[track_id][asset_col_name] = True
        all_possible_asset_cols.add(asset_col_name)

    final_rows = []
    # Ensure track_ids are sorted numerically
    sorted_track_ids = sorted(manifest_data.keys(), key=int)
    for track_id in sorted_track_ids:
        row = {"track_id": track_id}
        # Ensure consistent column order
        for col_name in sorted(list(all_possible_asset_cols)):
            row[col_name] = manifest_data[track_id].get(col_name, False)
        final_rows.append(row)

    if not final_rows:
        fieldnames = ["track_id"] + sorted(list(all_possible_asset_cols))
    else:
        fieldnames = list(final_rows[0].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nManifest with {len(final_rows)} tracks written to: {output_path}")
    print(f"Columns in manifest: {fieldnames}")


def main():
    parser = argparse.ArgumentParser(description="Generate a local SALAMI dataset manifest.")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root directory (e.g., ~/data/salami).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for metadata.csv (default: {dataset_root}/metadata.csv)",
    )
    args = parser.parse_args()

    # Expand user-provided paths
    dataset_root = args.dataset_root.expanduser()
    if not dataset_root.is_dir():
        print(f"Error: Dataset root not found: {dataset_root}")
        return 1

    output_path = (args.output or dataset_root / "metadata.csv").expanduser()

    try:
        build_manifest(dataset_root, output_path)
        return 0
    except Exception as e:
        print(f"Error building manifest: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
