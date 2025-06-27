#!/usr/bin/env python3
"""Generate metadata manifest for SALAMI dataset.

This script scans the SALAMI dataset directory structure and creates
a metadata.csv file that serves as the authoritative index of all
data assets.

Usage:
    python scripts/build_manifest.py ~/data/salami
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def find_audio_files(audio_dir: Path) -> List[Tuple[str, Path]]:
    """Find all audio files in the SALAMI audio directory structure.

    Returns:
        List of (track_id, audio_path) tuples
    """
    audio_files = []
    if not audio_dir.exists():
        print(f"Warning: Audio directory not found: {audio_dir}")
        return audio_files

    # SALAMI structure: audio/{track_id}/audio.{ext}
    for track_dir in audio_dir.iterdir():
        if not track_dir.is_dir() or not track_dir.name.isdigit():
            continue

        track_id = track_dir.name
        audio_extensions = [".mp3", ".wav", ".flac", ".m4a"]

        for ext in audio_extensions:
            audio_file = track_dir / f"audio{ext}"
            if audio_file.exists():
                audio_files.append((track_id, audio_file))
                break

    return audio_files


def find_jams_files(jams_dir: Path) -> List[Tuple[str, Path]]:
    """Find all JAMS annotation files.

    Returns:
        List of (track_id, jams_path) tuples
    """
    jams_files = []
    if not jams_dir.exists():
        print(f"Warning: JAMS directory not found: {jams_dir}")
        return jams_files

    # SALAMI structure: jams/{track_id}.jams
    for jams_file in jams_dir.glob("*.jams"):
        track_id = jams_file.stem
        if track_id.isdigit():  # Validate numeric track ID
            jams_files.append((track_id, jams_file))

    return jams_files


def build_salami_manifest(dataset_root: Path, output_path: Path) -> None:
    """Build metadata manifest for SALAMI dataset.

    Parameters:
        dataset_root: Path to SALAMI dataset root directory
        output_path: Path where to write the metadata.csv file
    """
    audio_dir = dataset_root / "audio"
    jams_dir = dataset_root / "jams"

    print(f"Scanning SALAMI dataset at: {dataset_root}")

    # Collect all data assets
    audio_files = find_audio_files(audio_dir)
    jams_files = find_jams_files(jams_dir)

    print(f"Found {len(audio_files)} audio files")
    print(f"Found {len(jams_files)} JAMS files")

    # Build manifest records
    records = []

    # Add audio assets
    for track_id, audio_path in audio_files:
        records.append(
            {
                "track_id": track_id,
                "asset_type": "audio",
                "asset_subtype": "mix",
                "file_path": str(audio_path),
            }
        )

    # Add annotation assets
    for track_id, jams_path in jams_files:
        records.append(
            {
                "track_id": track_id,
                "asset_type": "annotation",
                "asset_subtype": "reference",
                "file_path": str(jams_path),
            }
        )

    # Sort by track_id for consistent ordering
    records.sort(key=lambda x: int(x["track_id"]))

    # Write manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = ["track_id", "asset_type", "asset_subtype", "file_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Manifest written to: {output_path}")
    print(f"Total records: {len(records)}")

    # Basic validation
    track_ids = {r["track_id"] for r in records}
    print(f"Unique tracks: {len(track_ids)}")

    audio_tracks = {r["track_id"] for r in records if r["asset_type"] == "audio"}
    jams_tracks = {r["track_id"] for r in records if r["asset_type"] == "annotation"}

    missing_audio = jams_tracks - audio_tracks
    missing_jams = audio_tracks - jams_tracks

    if missing_audio:
        print(f"Warning: {len(missing_audio)} tracks have JAMS but no audio")
    if missing_jams:
        print(f"Warning: {len(missing_jams)} tracks have audio but no JAMS")


def main():
    parser = argparse.ArgumentParser(description="Generate SALAMI dataset manifest")
    parser.add_argument(
        "dataset_root", type=Path, help="Path to SALAMI dataset root directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for metadata.csv (default: {dataset_root}/metadata.csv)",
    )

    args = parser.parse_args()

    if not args.dataset_root.exists():
        print(f"Error: Dataset root not found: {args.dataset_root}")
        return 1

    if args.output is None:
        args.output = args.dataset_root / "metadata.csv"

    try:
        build_salami_manifest(args.dataset_root, args.output)
        return 0
    except Exception as e:
        print(f"Error building manifest: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
