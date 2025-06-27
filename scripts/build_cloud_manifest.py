#!/usr/bin/env python3
"""
Cloud-native metadata manifest builder for datasets hosted on R2.

This script scans an R2 bucket and creates a manifest file with full asset URLs.

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
        "asset_key": "audio_mp3_path",
        "path_pattern": r"slm-dataset/(?P<track_id>\d+)/audio\.mp3",
    },
    {
        "asset_key": "annotation_reference_path",
        "path_pattern": r"ref-jams/(?P<track_id>\d+)\.jams",
    },
    {
        "asset_key": "annotation_adobe-mu1gamma1_path",
        "path_pattern": r"adobe21-est/def_mu_0\.1_gamma_0\.1/(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json",
    },
    {
        "asset_key": "annotation_adobe-mu5gamma5_path",
        "path_pattern": r"adobe21-est/def_mu_0\.5_gamma_0\.5/(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json",
    },
    {
        "asset_key": "annotation_adobe-mu1gamma9_path",
        "path_pattern": r"adobe21-est/def_mu_0\.1_gamma_0\.9/(?P<track_id>\d+)\.mp3\.msdclasscsnmagic\.json",
    },
]


def get_all_r2_object_keys() -> list[str]:
    """Authenticate with R2 and retrieve all object keys from the bucket."""
    try:
        import boto3
    except ImportError:
        print("Error: boto3 is required. Install with: pip install boto3")
        raise

    if not all([ACCOUNT_ID, ACCESS_KEY_ID, SECRET_ACCESS_KEY]):
        raise ValueError(
            "Missing R2 credentials. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "and R2_SECRET_ACCESS_KEY environment variables."
        )

    print(f"Connecting to R2 bucket: {BUCKET_NAME}...")
    s3 = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="auto",
    )

    all_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME):
        if "Contents" in page:
            keys = [obj["Key"] for obj in page["Contents"]]
            all_keys.extend(keys)
            print(f"Retrieved {len(keys)} object keys...")
    print(f"Found {len(all_keys)} total objects in bucket.")
    return all_keys


def process_paths(all_paths: list[str], output_path: Path):
    """Process a list of cloud paths and create the manifest with full URLs."""
    assets_by_track = {}
    all_asset_keys = {spec["asset_key"] for spec in ASSET_SPECS}

    for path_str in all_paths:
        for spec in ASSET_SPECS:
            match = re.search(spec["path_pattern"], path_str)
            if match:
                track_id = match.group("track_id")
                assets_by_track.setdefault(track_id, {"track_id": track_id})
                asset_url = f"{PUBLIC_URL_BASE}/{path_str}"
                assets_by_track[track_id][spec["asset_key"]] = asset_url
                break  # Move to next path once a match is found

    if not assets_by_track:
        print("Warning: No assets matched the specs. Manifest will be empty.")
        return

    manifest_df = pd.DataFrame(assets_by_track.values())

    # Ensure all potential columns are present, filling missing with None (or pd.NA)
    for key in all_asset_keys:
        if key not in manifest_df.columns:
            manifest_df[key] = pd.NA

    # Sort columns alphabetically, with track_id first
    sorted_columns = ["track_id"] + sorted([col for col in manifest_df.columns if col != "track_id"])
    manifest_df = manifest_df[sorted_columns]

    # Sort by track_id numerically
    manifest_df["track_id"] = pd.to_numeric(manifest_df["track_id"])
    manifest_df = manifest_df.sort_values("track_id").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    print(f"\nManifest with {len(manifest_df)} tracks written to: {output_path}")
    print(f"Columns: {list(manifest_df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Build a cloud dataset manifest with full asset URLs from R2.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("slm_manifest_cloud.csv"),
        help="Output path for the manifest CSV file.",
    )
    args = parser.parse_args()

    try:
        all_keys = get_all_r2_object_keys()
        process_paths(all_keys, args.output.expanduser())
    except (ImportError, ValueError) as e:
        print(f"Setup Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
