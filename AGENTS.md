# Agent Instructions and Preferences

This document provides guidance and preferences for AI agents working with this codebase.

## General Principles

- **Lean, Focused, and Healthy Codebase:** Strive to keep the codebase concise, well-organized, and maintainable. Avoid unnecessary complexity and ensure code is easy to understand and test.
- **Test-Driven Development:** When practical, write tests before implementing new features or fixing bugs. Tests are crucial for verifying correctness and preventing regressions.
- **Clear Documentation:** Document code clearly, especially for complex logic or public APIs. Use docstrings and comments where appropriate.

## Specific Preferences

- **Hierarchy Plotting:** When adding new plotting methods for `Hierarchy` objects, consider both multi-axes (one per layer) and single-axis (all layers on one plot with visual distinction) representations. The single-axis plot is useful for comparing layer alignments directly.
- **JSON Conversion:** For `Hierarchy.from_json`, the expected input format is a list of layers, where each layer is a list of segments, and each segment is a tuple of `(start_time, end_time, label_str)`.

## Future Considerations (Agent Notes)

* (Agent can add notes here as development progresses, e.g., areas for refactoring, common pitfalls, useful commands, etc.)
