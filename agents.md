# Agent Instructions and Preferences

This document provides guidance and preferences for AI agents working with this codebase.

## General Principles

- **Lean, Focused, and Healthy Codebase:** Strive to keep the codebase concise, well-organized, and maintainable. Avoid unnecessary complexity and ensure code is easy to understand and test.
- **Test Strategy:** When practical, keep tests up to date, without slowing down development.
- **Clear Documentation:** Document code clearly, especially for complex logic or public APIs. Use docstrings and comments where appropriate, and sphinx-ify the codebase for RTD.

## Specific Preferences

- **Use Core Classes:** This repo is built around the core classes `TimeSeries`, `Segmentation`, and `Hierarchy`. Use them as much as possible, and keep them robust.
- **JSON Conversion:** For `Hierarchy.from_json`, the expected input format is a list of layers, where each layer is a list of segments, and each segment is a tuple of `(start_time, end_time, label_str)`.

## Future Considerations (Agent Notes)

* (Agent can add notes here as development progresses, e.g., areas for refactoring, common pitfalls, useful commands, etc.)
