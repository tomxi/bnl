# BNL: Boundaries and Labels

A Python library for hierarchical Music Structure Analysis with cloud-native dataset management.

> This repo is co-authored with LLMs.

## Install and Develop

```bash
git clone https://github.com/tomxi/bnl.git
cd bnl
pip install -e '.[dev,docs,dash,notebooks]'  # Install all core and optional dependencies
```

## Core Concepts & Usage

The library provides a flexible pipeline for transforming raw, potentially non-monotonic hierarchies `MultiSegmentation` into clean, `BoundaryHierarchy` objects. This is achieved through a fluent, strategy-based API.

```python
import bnl

slm_ds = bnl.Dataset(manifest_path="~/data/salami/metadata.csv")
track = slm_ds[8]

ref = list(track.refs.values())[0]
est = list(track.ests.values())[0]

ref.plot().show()
est.plot().show()
```
### Configuration (optional)

You can customize data sources and HTTP behavior via environment variables:

- `BNL_R2_BUCKET_PUBLIC_URL`: override the default public bucket base URL.
- `BNL_HTTP_TIMEOUT`: request timeout in seconds (default: 10).

## Development

```bash
pytest  # Run tests
mypy src/bnl  # Check types
ruff format src/bnl  # Format code
ruff check src/bnl  # Lint code
ruff check src/bnl --fix # Fix linting issues
sphinx-build -b html docs build/html  # Build docs
sphinx-autobuild docs build/html --port 8000 # Serve docs with hot reload
```