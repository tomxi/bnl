#!/usr/bin/env python3
"""
Cloud-native metadata manifest builder for datasets hosted on R2.

This script scans an R2 bucket and creates a boolean manifest file (has_* columns)
that is compatible with the BNL data loading system.

Usage:
python scripts/build_cloud_manifest.py [--track-ids 1,2,3]
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

# --- R2 Configuration ---
BUCKET_NAME = "slm-dataset"
PUBLIC_URL_BASE = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev"  # Found in bucket settings

# R2 Authentication (optional - set as environment variables or uncomment and fill)
ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")  # Your Cloudflare account ID
ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")  # Your R2 access key ID
SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")  # Your R2 secret access key

# --- Asset Specifications ---
# Updated to match your actual folder structure and create boolean columns
ASSET_SPECS: list[dict[str, Any]] = [
    {
        "asset_type": "audio",
        "asset_subtype": "mp3",
        "path_pattern": r"slm-dataset/.*\.mp3$",  # Audio files in slm-dataset folder
        "track_id_pattern": r"slm-dataset/(?P<track_id>\d+)/audio\.mp3",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "reference",  # Changed from "ref" to "reference" to match expected boolean format
        "path_pattern": r"ref-jams/.*\.jams$",  # Reference JAMS files
        "track_id_pattern": r"ref-jams/(?P<track_id>\d+)\.jams",
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "adobe-mu1gamma1",
        "path_pattern": (r"adobe21-est/def_mu_0\.1_gamma_0\.1/.*\.mp3\.msdclasscsnmagic\.json$"),
        "track_id_pattern": (
            r"adobe21-est/def_mu_0\.1_gamma_0\.1/"
            r"(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json"
        ),
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "adobe-mu5gamma5",
        "path_pattern": (r"adobe21-est/def_mu_0\.5_gamma_0\.5/.*\.mp3\.msdclasscsnmagic\.json$"),
        "track_id_pattern": (
            r"adobe21-est/def_mu_0\.5_gamma_0\.5/"
            r"(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json"
        ),
    },
    {
        "asset_type": "annotation",
        "asset_subtype": "adobe-mu1gamma9",
        "path_pattern": (r"adobe21-est/def_mu_0\.1_gamma_0\.9/.*\.mp3\.msdclasscsnmagic\.json$"),
        "track_id_pattern": (
            r"adobe21-est/def_mu_0\.1_gamma_0\.9/"
            r"(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json"
        ),
    },
]


def build_cloud_manifest_authenticated(output_path: Path):
    """Builds a manifest by listing objects in an R2 bucket using authentication."""
    try:
        import boto3
    except ImportError:
        print("Error: boto3 is required for authenticated mode. Install with: pip install boto3")
        return False

    if not all([ACCOUNT_ID, ACCESS_KEY_ID, SECRET_ACCESS_KEY]):
        print(
            "Error: Missing R2 credentials. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "and R2_SECRET_ACCESS_KEY environment variables."
        )
        return False

    print(f"Connecting to R2 bucket: {BUCKET_NAME}")

    s3 = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="auto",
    )

    try:
        # Handle pagination to get ALL objects
        all_objects = []
        continuation_token = None

        while True:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=BUCKET_NAME, ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=BUCKET_NAME)

            # Add objects from this page
            objects = response.get("Contents", [])
            all_objects.extend(objects)
            print(f"Retrieved {len(objects)} objects (total so far: {len(all_objects)})")

            # Check if there are more objects
            if response.get("IsTruncated", False):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

        all_paths = [obj["Key"] for obj in all_objects]
        print(f"Found {len(all_paths)} total objects in bucket.")

        return process_paths(all_paths, output_path)
    except Exception as e:
        print(f"Error connecting to R2: {e}")
        return False


def build_cloud_manifest_from_existing():
    """Build a manifest from the existing manifest to get all track IDs."""
    print("Fetching existing manifest to get track IDs...")

    try:
        import requests

        response = requests.get(f"{PUBLIC_URL_BASE}/manifest_cloud.csv")
        response.raise_for_status()

        import io

        existing_df = pd.read_csv(io.StringIO(response.text))
        track_ids = existing_df["track_id"].astype(str).tolist()

        print(f"Found {len(track_ids)} track IDs in existing manifest")
        return track_ids
    except Exception as e:
        print(f"Error fetching existing manifest: {e}")
        return None


def build_cloud_manifest_from_track_ids(track_ids: list[str], output_path: Path):
    """Builds a manifest from known track IDs without authentication."""
    print(f"Building boolean manifest for {len(track_ids)} track IDs")

    # Generate expected paths based on track IDs and ASSET_SPECS
    all_paths = []
    for track_id in track_ids:
        # Add expected paths for each asset type
        all_paths.extend(
            [
                f"slm-dataset/{track_id}/audio.mp3",
                f"ref-jams/{track_id}.jams",
                f"adobe21-est/def_mu_0.1_gamma_0.1/{track_id}.mp3.msdclasscsnmagic.json",
                f"adobe21-est/def_mu_0.5_gamma_0.5/{track_id}.mp3.msdclasscsnmagic.json",
                f"adobe21-est/def_mu_0.1_gamma_0.9/{track_id}.mp3.msdclasscsnmagic.json",
            ]
        )

    print(f"Generated {len(all_paths)} expected paths.")
    return process_paths(all_paths, output_path)


def _build_assets_by_track(all_paths: list[str], asset_specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Builds a dictionary of assets grouped by track ID."""
    assets_by_track: dict[str, dict[str, Any]] = {}
    for spec in asset_specs:
        path_pattern = re.compile(spec["path_pattern"])
        track_id_pattern = re.compile(spec["track_id_pattern"])
        matched_paths = 0
        for path_str in all_paths:
            if path_pattern.match(path_str):
                match = track_id_pattern.search(path_str)
                if match:
                    track_id = match.group("track_id")
                    matched_paths += 1
                    assets_by_track.setdefault(track_id, {"track_id": track_id})
                    col_name = f"has_{spec['asset_type']}_{spec['asset_subtype']}"
                    assets_by_track[track_id][col_name] = True
        print(f"  {spec['asset_type']}_{spec['asset_subtype']}: {matched_paths} files matched")
    return assets_by_track


def _create_manifest_df(
    assets_by_track: dict[str, dict[str, Any]], asset_specs: list[dict[str, Any]]
) -> pd.DataFrame | None:
    """Creates a manifest DataFrame from assets grouped by track."""
    if not assets_by_track:
        print("Warning: No assets matched the specs. Manifest will be empty.")
        return None

    all_possible_asset_cols = {f"has_{spec['asset_type']}_{spec['asset_subtype']}" for spec in asset_specs}

    records = []
    for track_id, assets in assets_by_track.items():
        record = {"track_id": track_id}
        record.update({col: assets.get(col, False) for col in all_possible_asset_cols})
        records.append(record)

    manifest_df = pd.DataFrame(records)
    sorted_columns = ["track_id"] + sorted(list(all_possible_asset_cols))

    for col in sorted_columns:
        if col not in manifest_df.columns:
            manifest_df[col] = False

    manifest_df = manifest_df[sorted_columns]
    manifest_df.sort_values(by="track_id", key=lambda col: col.astype(int), inplace=True)
    return manifest_df


def _save_and_summarize_manifest(manifest_df: pd.DataFrame, output_path: Path):
    """Saves the manifest DataFrame to CSV and prints a summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    print(f"\nBoolean manifest with {len(manifest_df)} tracks written to: {output_path}")
    print(f"Columns in manifest: {list(manifest_df.columns)}")
    print("\nAsset availability summary:")
    asset_cols = [col for col in manifest_df.columns if col != "track_id"]
    for col in asset_cols:
        count = manifest_df[col].sum()
        percentage = (count / len(manifest_df)) * 100
        print(f"  {col}: {count}/{len(manifest_df)} tracks ({percentage:.1f}%)")


def process_paths(all_paths: list[str], output_path: Path) -> bool:
    """Process a list of paths and create the boolean manifest."""
    assets_by_track = _build_assets_by_track(all_paths, ASSET_SPECS)
    manifest_df = _create_manifest_df(assets_by_track, ASSET_SPECS)

    if manifest_df is None:
        print("Warning: Manifest DataFrame could not be created.")
        return False

    _save_and_summarize_manifest(manifest_df, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate a boolean cloud dataset manifest.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("manifest_cloud_boolean.csv"),
        help="Output path for manifest CSV (default: manifest_cloud_boolean.csv)",
    )
    parser.add_argument(
        "--track-ids",
        type=str,
        help=(
            "Comma-separated list of track IDs (e.g., '1,2,3'). "
            "If not provided, will use existing manifest or try authenticated bucket listing."
        ),
    )

    args = parser.parse_args()

    try:
        if args.track_ids:
            # Use provided track IDs
            track_ids = [tid.strip() for tid in args.track_ids.split(",")]
            success = build_cloud_manifest_from_track_ids(track_ids, args.output)
        else:
            # Try to get track IDs from existing manifest first
            track_ids = build_cloud_manifest_from_existing()
            if track_ids:
                success = build_cloud_manifest_from_track_ids(track_ids, args.output)
            else:
                # Fall back to authenticated mode - list bucket contents
                print("Falling back to authenticated bucket listing...")
                success = build_cloud_manifest_authenticated(args.output)

        if success:
            print(f"\n✅ Boolean manifest created successfully: {args.output}")
            print("\nTo upload this manifest to your R2 bucket, run:")
            print(f"  rclone copy {args.output} r2-bnl:{BUCKET_NAME}")
            print("\nTo use this boolean format in your app, update the Dataset constructor:")
            print("  # Change data_source_type or use the boolean manifest URL")
            return 0
        else:
            print("❌ Failed to create manifest")
            return 1

    except Exception as e:
        print(f"Error building manifest: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
