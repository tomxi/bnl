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

### User is working on...
- Refactoring the core classes to decouple Boundary and Label.

### User Feedbacks
- the from_jams and from_json methods are still not working... we need to fix them.
- hmm I need a way to quary the label at a timepoint, or the closest boundary in past time basically.
- I want to decouple BOundary and label again... see the API Handoff Summary for the latest design.
Of course. Here is the API design in markdown format.

### **Core API Design for Monotonic Boundary Casting**

**Guiding Principle**: The design separates pure structural markers (`Boundary`) from their descriptive metadata (`labels`). Labels are managed by the container objects in which the boundaries frame. This allows the core transformation pipeline to operate purely on structure.

---
### **Key Methods & Research Pipeline**

The API is designed to support the following workflow for investigating boundary monotonicity.

* **Core Transformation Pipeline**:
    1.  `MultiSegment.to_contour() -> BoundaryContour`
        * This method aggregates boundaries from all layers of a `MultiSegment` into a single salience contour.
    2.  `BoundaryContour.to_levels() -> BoundaryHierarchy`
        * This is the core algorithmic step. It takes the salience contour and generates a set of leveled boundaries that adhere to the principle of monotonicity.

* **Plotting**:
    * `MultiSegment.plot()`: The main, user-facing plotting method to visualize the input annotations.

---
### **Plotting Architecture 🎨**

Plotting is handled hierarchically to ensure visual consistency.

* **`TimeSpan.plot()`**: The fundamental drawing method. It plots a single named, colored span (`axvspan`) based on a provided styling context.
* **`Segment.plot()`**: Composes a plot by calling `.plot()` on each of its internal `TimeSpan` sections.
* **`MultiSegment.plot()`**: The top-level entry point. It is responsible for creating the styling context and orchestrating the plotting of all its `Segment` layers.

#### **The `PlottingStyleMaps` Mechanism**

1.  **Context Creation**: Before drawing, `MultiSegment.plot()` scans all components to pre-build a `PlottingStyleMaps` object. This object maps every unique label to a consistent style (e.g., `'Verse'` → `'C0'`).
2.  **Context Passing**: This `PlottingStyleMaps` object is passed down as an internal argument through the call stack (`MultiSegment` → `Segment` → `TimeSpan`).
3.  **Coordinated Drawing**: Each `TimeSpan.plot()` method uses the received style map to look up the correct style for its label, ensuring a globally consistent and readable plot.