# Agent Instructions

Concise guidance for AI agents working with this music information retrieval codebase.

## Core Principles

- **Simplicity First:** Keep code elegant, concise, and maintainable. Avoid over-engineering. Cleanliness is godliness.
- **Core Classes:** Built around `TimeSeries`, `Segmentation`, and `Hierarchy` - use extensively and keep robust
- **Thoughtful API Design:** Design the API for the core classes to be intuitive and easy to use; consider the user's perspective; this is tooling for researchers, not for developers;
- **Research Focus:** Support music structure analysis and representation research questions
- **80/20 Rule:** Achieve maximum impact with minimal effort

## Development Guidelines

- **Testing:** Maintain tests pragmatically without slowing development; make tests failure driven; test apps/ folder too and add tests from app failures;
- **Documentation:** Clear docstrings for complex logic and public APIs; Sphinx + RTD ready; build and consult the docs actively during development, planning, and ideation
- **Dependencies:** Prefer pixi; minimal external dependencies; keep tomls current and use pixi tasks.
    - **Pixi Tasks:** Use `pixi run check` for a full validation pipeline (format, lint, types, test). Other tasks like `pixi run format`, `pixi run lint`, `pixi run test`, `pixi run docs-build` are also available. Consult `pixi.toml` for details; keep toml files current and run `pixi install` when toml changes.
- **Code Style:** Meaningful naming, logical structure, selective type hints. Adhere to ruff formatting.
- **Data Format:** Prefer feather files; prioritize native core classes over jams/json/mir_eval formats but support them.


## Agent Notes

*(Add development insights, common patterns, useful commands, etc.)*

### User is working on...
- building the baseline for monotonic casting
- cleaning up the api to keep the mental model simple and codebase under control.

### User Feedbacks
- plotting is not using the old styledict management... we used to make surethat the same label all gets the same stylemap, even if they are across different layers. We had a a stylemap generator that was specicallyv dealing with generating the style maps given a list of labels of the hierarchy.
- Relatedly, the plot2 method is suppose to make that management much easier... since we will be plotting all using the same axis so it can just use its own style dict etc...? Regardless, we need to make a issue or find an exisiting one and append to it?

### API Decisions
- `TimeSpan` now uses `(start: Boundary, duration: float)` for initialization instead of `(start: Boundary, end: Boundary)`. The `end` attribute is now a calculated property. This simplifies instantiation and validation.
- **Monotonic Casting Pipeline Refactored:** The synthesis process now uses a fluent, strategy-based pipeline.
    - **Core Idea:** `Hierarchy` -> `RatedBoundaries` -> `ProperHierarchy`.
    - **`RatedBoundaries`:** An intermediate, chainable class (`.group_boundaries()`, `.quantize_level()`).
    - **`bnl.ops.Pipeline`:** The main entry point that orchestrates the process.
    - **`bnl.strategies`:** Defines contracts for `SalienceStrategy`, `BoundaryGroupingStrategy`, and `LevelGroupingStrategy`. This allows for pluggable algorithms at each stage of the synthesis.
    - **Logic Placement:** Synthesis logic (creating layers from rated events) now lives in strategies (e.g., `DirectSynthesisStrategy`), not in the core data classes.