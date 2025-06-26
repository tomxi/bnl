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
