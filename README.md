# BNL

Bounaries & Labels (BNL)
Co-written with LLMs via Windsurf.

## Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Basic Usage

```python
from bnl import Segment, Hierarchy

# Create segments
segment_a = Segment(0.0, 2.5, "A")
segment_b = Segment(2.5, 5.0, "B")

# Create a hierarchy
hierarchy = Hierarchy(name="example")
hierarchy.add_level([segment_a, segment_b])

print(hierarchy)  # Hierarchy(example, levels=[Level 0: 2 segments])
```

## Development

Run tests:
```bash
pytest tests/
```
