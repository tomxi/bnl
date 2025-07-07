# Agent Instructions

Concise guidance for AI agents working with this music information retrieval codebase.

## Core Principles

- **Simplicity First:** Keep code elegant, concise, and maintainable. Avoid over-engineering. Cleanliness is godliness.
- **Core Classes:** Built around classes in `bnl/core/` - use extensively and keep robust
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
- Don't attempt to edit notebooks; feel free to read them, but don't create or edit them.
### User is working on...
- Get the guassian convoled Boundary Saliecne Curve working

### User Feedbacks
### **Key Methods & Research Pipeline**

The API is designed to support the following workflow for investigating boundary monotonicity.

* **Core Transformation Pipeline**:
`MultiSegment -> BoundaryContour -> BoundaryHierarchy`
- We'll just use `ops` to operate on the `MultiSegment` directly.


### Lessons from Refactoring `ops.py`

- **Trust the Data Models:** The core data classes in `bnl.core` (e.g., `BoundaryHierarchy`, `Segment`) have validation built-in. For example, a `BoundaryHierarchy` must have at least two boundaries. Functions operating on these objects can and should *assume* they are well-formed. Avoid writing defensive checks for conditions the data model already prevents.
- **Clarity Over Cleverness:** A straightforward, readable implementation (e.g., a simple loop or a generator function) is superior to a complex one-liner that is difficult to debug and understand.
- **Efficient Hierarchy Construction:** When transforming a `BoundaryHierarchy` into a `MultiSegment`, the most efficient strategy is to first group boundaries by level (e.g., into a dictionary) and then iterate through the levels, accumulating boundaries. This is much better than re-filtering the entire list of boundaries for each level.