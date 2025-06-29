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
