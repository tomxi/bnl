# BNL: Boundaries and Labels

A Python library for hierarchical Music Structure Analysis with cloud-native dataset management.

> This repo is co-authored with LLMs.

## Install and Develop

```bash
git clone https://github.com/tomxi/bnl.git
cd bnl
pixi install
pixi run pip install -e .[dev,docs]
```

## Core Concepts & Usage

The library provides a flexible pipeline for transforming raw, potentially non-monotonic hierarchies into clean, `ProperHierarchy` objects. This is achieved through a fluent, strategy-based API.

```python
import bnl
from bnl import strategies

# 1. Load a raw hierarchy (e.g., from a JSON file)
h = bnl.Hierarchy.from_json(...)

# 2. Configure the processing pipeline with desired strategies
pipeline = bnl.Pipeline(
    salience_strategy=strategies.FrequencyStrategy(),
    grouping_strategy=None, # Optional: add a BoundaryGroupingStrategy here
    leveling_strategy=strategies.DirectSynthesisStrategy(),
)

# 3. Process the hierarchy to get a guaranteed-monotonic result
proper_h = pipeline.process(h)
```

## Interactive Data Explorer

Launch the Streamlit app to explore the dataset:
```bash
pixi run exp
```

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
├── adobe/<hyper_parameter>/<track_id>.mp3msdscnclassmagicsome_other_string.json
└── metadata.csv (generated)
```

## Development

```bash
pixi run test
pixi run types
pixi run format
pixi run check
pixi run fix
pixi run docs-build  # Build docs
pixi run docs-serve  # Serve with hot reload
```

## Deployment

### Streamlit Community Cloud

The project includes a `requirements.txt` file specifically for Streamlit Community Cloud hosting. While local development uses pixi for dependency management, Streamlit Cloud requires a traditional requirements.txt file.

> **Note**: Do not delete `requirements.txt` - it's essential for cloud deployment, even though we use pixi locally.
