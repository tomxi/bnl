# BNL: Boundaries and Labels

A Python library for hierarchical Music Structure Analysis with cloud-native dataset management.

> This repo is co-authored with LLMs.

## Quick Start

```bash
git clone https://github.com/tomxi/bnl.git
cd bnl
pixi install
pixi run pip install -e .[dev,docs]
```

## Core Functionality

```python
import bnl

# Load cloud dataset
dataset = bnl.data.Dataset("https://pub-<username>.r2.dev/manifest_cloud_boolean.csv")

# Load a track by ID (ensure "2" is in dataset.track_ids)
track = dataset["2"]
print(f"Track info: {track.info.get('title', 'N/A')} by {track.info.get('artist', 'N/A')}")

# Load audio
waveform, sr = track.load_audio()
if waveform is not None:
    print(f"Loaded audio: {waveform.shape}, SR: {sr}")

# Load 'reference' annotation if available
if "reference" in track.annotations:
    try:
        annotation = track.load_annotation("reference")
        print(f"Loaded reference annotation: {type(annotation)}")
        # You can now plot or analyze the annotation (bnl.Hierarchy or bnl.Segmentation)
        # For example, if it's a Hierarchy:
        # if isinstance(annotation, bnl.core.Hierarchy):
        #     fig = annotation.plot()
        #     fig.show() # Requires a graphical backend
    except Exception as e:
        print(f"Error loading reference annotation: {e}")
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
pixi run build-local-manifest -- ~/data/salami
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

## Deployment

### Streamlit Community Cloud

The project includes a `requirements.txt` file specifically for Streamlit Community Cloud hosting. While local development uses pixi for dependency management, Streamlit Cloud requires a traditional requirements.txt file.

> **Note**: Do not delete `requirements.txt` - it's essential for cloud deployment, even though we use pixi locally.

## License

MIT
