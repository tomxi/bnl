# BNL: Boundaries and Labels

A Python library for hierarchical Music Structure Analysis with cloud-native dataset management.

> This repo is co-authored with LLMs.

## Install and Develop

```bash
git clone https://github.com/tomxi/bnl.git
cd bnl
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,docs,dash,notebooks]'  # Install all core and optional dependencies
```

## Core Concepts & Usage

The library provides a flexible pipeline for transforming raw, potentially non-monotonic hierarchies into clean, `ProperHierarchy` objects. This is achieved through a fluent, strategy-based API.

```python
import bnl

slm_ds = bnl.data.Dataset(manifest_path="~/data/salami/metadata.csv")
track = slm_ds[8]

ref = track.load_annotation("reference")
est = track.load_annotation("adobe-mu1gamma1")

ref.plot().show()
est.plot().show()
```
## Development

```bash
uv run pytest  # Run tests
uv run mypy src/bnl  # Check types
uv run ruff format src/bnl  # Format code
uv run ruff check src/bnl  # Lint code
uv run ruff check src/bnl --fix # Fix linting issues
uv run sphinx-build -b html docs build/html  # Build docs
uv run sphinx-autobuild docs build/html --port 8000 # Serve docs with hot reload
```