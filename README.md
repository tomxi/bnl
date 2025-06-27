# BNL: Boundaries and Labels

A Python library for hierarchical Music Structure Analysis with cloud-native dataset management.

> This repo is co-authored with LLMs.

## Quick Start

```bash
git clone https://github.com/tomxi/bnl.git
cd bnl
pixi install
```

## Core Functionality

```python
import bnl

# Load cloud dataset (default)
dataset = bnl.data.Dataset("https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/manifest_cloud_boolean.csv",
                           data_source_type="cloud",
                           cloud_base_url="https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev")

# Load a track with metadata and audio
track = dataset.load_track("2")
print(f"Now playing: \"{track.info['title']}\" by {track.info['artist']}")

waveform, sr = track.load_audio()  # Load MP3 from cloud
```

## Interactive Data Explorer

Launch the Streamlit app to explore the dataset:

```bash
pixi run exp
```

The app provides:
- **Track browser**: Browse 1,400+ music tracks with metadata
- **Audio playback**: Stream MP3s directly from cloud storage  
- **Visualization**: Waveform and MFCC analysis
- **Dual mode**: Cloud (default) or local dataset support

Access at: http://localhost:8502

## Dataset Management

### Cloud Datasets (Recommended)

Cloud datasets use boolean manifests with automatic URL reconstruction:

```bash
# Generate boolean manifest from existing cloud data
pixi run build-cloud-manifest
```

### Local Datasets

For local SALAMI datasets:

```bash
# Generate manifest for local dataset
pixi run build-manifest -- ~/data/salami
```

Expected structure:
```
dataset_root/
├── audio/<track_id>/audio.mp3
├── jams/<track_id>.jams
└── metadata.csv (generated)
```

## Development

Core tasks:
```bash
pixi run format    # Format code
pixi run check     # Lint code  
pixi run test      # Run tests
pixi run types     # Type check
```

Documentation:
```bash
pixi run docs-build  # Build docs
pixi run docs-serve  # Serve with hot reload
```

## License

MIT
