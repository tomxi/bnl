# BNL: Boundaries and Labels

A Python library for working with hierarchical Music Structure Analysis.

> This repo is co-authored with LLMs in their various forms.

## Quick Start

```python
import numpy as np
from bnl import Segment, plot_segment

# Create a segment
boundaries = {0.0, 2.5, 5.0, 7.5, 10.0}  # Set of boundary times
labels = ['A', 'B', 'A', 'C']
seg = Segment(boundaries, labels)

# Visualize
fig, ax = plot_segment(seg, text=True)
```

## Installation & Setup

This project is managed by [Pixi](https://pixi.sh/).

```bash
git clone https://github.com/tomxi/bnl.git
cd bnl
pixi install
```

This command installs all dependencies (including development tools) into a local `.pixi` environment and makes the `bnl` package available for use. If you encounter issues with the `bnl` module not being found during testing, ensure your project is installed in editable mode (e.g., by running `pip install -e .` within the pixi environment if `pixi install` doesn't handle it automatically, though typically it should for projects with `pyproject.toml`).

## Data Management

The data loading system relies on a central `metadata.csv` manifest file to index all dataset assets. This file must be generated before you can load data.

### Building the Manifest

A flexible script is provided to scan your dataset directory and create the manifest.

1.  **Organize your data**: Ensure your dataset (e.g., SALAMI) is stored in a structured way. The default configuration expects a layout like:
    ```
    <dataset_root>/
    ├── audio/
    │   └── <track_id>/
    │       └── audio.mp3
    └── jams/
        └── <track_id>.jams
    ```

2.  **Run the script**: Point the script to your dataset's root directory.

    ```bash
    pixi run build-manifest -- <path_to_your_dataset_root>
    ```

    For example, if your SALAMI dataset is in `~/data/salami`:
    ```bash
    pixi run build-manifest -- ~/data/salami
    ```

This will create a `metadata.csv` file inside that directory, which can then be used by the data loaders. The script is configurable within `scripts/build_manifest.py` to support other datasets and file layouts.

## Development

All development tasks are managed via `pixi run <task_name>`. See `pixi.toml` for the full list of tasks.

Key tasks include:
- `pixi run format`: Auto-formats all code using Ruff.
- `pixi run fix`: Lints and auto-fixes safe violations using Ruff.
- `pixi run check`: Verifies formatting and linting (Ruff).
- `pixi run test`: Runs all tests using pytest.
- `pixi run types`: Runs mypy for static type analysis.
- `pixi run test-cov`: Runs tests with code coverage.
- `pixi run docs-build`: Builds Sphinx documentation.
- `pixi run docs-serve`: Serves documentation with live-reloading.

For a comprehensive check (format, lint, types, tests), you can run the individual commands or set up a combined task in `pixi.toml` if needed. The `pixi.toml` currently defines `check` for format and lint, and `test` for tests. `types` is separate.
To run all critical checks, execute sequentially:
```bash
pixi run format
pixi run check
pixi run types
pixi run test
```

You can see all available tasks by simply running `pixi run` in your terminal.

## Features

- **Core**: Hierarchical text segmentation with `Segment` class
- **Visualization**: Rich plotting with customizable styling  
- **Integration**: Compatible with mir_eval and MIR tools

## License

MIT
