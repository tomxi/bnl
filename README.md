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
pixi run install-dev
```

This command installs all dependencies into a local `.pixi` environment and makes the `bnl` package available for use.

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

All development tasks are managed via `pixi run`.

- **Format code**: `pixi run format-all`
- **Run all checks (format, lint, types, tests)**: `pixi run check-all`
- **Serve docs (live-reload)**: `pixi run docs-serve`
- **Run tests with coverage**: `pixi run test-cov`
- **See all available tasks**: `pixi run`

## Features

- **Core**: Hierarchical text segmentation with `Segment` class
- **Visualization**: Rich plotting with customizable styling  
- **Integration**: Compatible with mir_eval and MIR tools

## License

MIT
