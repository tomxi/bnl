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
#### **I. Point-like Objects 📍**

These are the fundamental, label-less structural markers on the timeline.

* **`Boundary`**
    * **Inherits From**: None (Base Class)
    * **Purpose**: A raw, unannotated marker on a timeline.
    * **Attributes**: `time: float`

* **`RatedBoundary`**
    * **Inherits From**: `Boundary`
    * **Purpose**: A boundary with a continuous measure of importance or salience.
    * **Attributes**: `salience: float`

* **`LeveledBoundary`**
    * **Inherits From**: `RatedBoundary`
    * **Purpose**: A definitive structural node within a monotonic hierarchy, identified by its time and discrete level.
    * **Attributes**: `level: int` (must be a positive integer)
    * **Implementation Note**: The constructor should enforce that `level` is a positive integer and automatically set the inherited `salience` attribute to be equal to the `level`.

---
#### **II. Span-like Objects (Containers) 🌊**

These objects represent time intervals and contain the point-like objects and their associated labels.

* **`TimeSpan`**
    * **Inherits From**: None (Base Class)
    * **Purpose**: Represents a generic time interval.
    * **Attributes**:
        * `start: Boundary`
        * `end: Boundary`
        * `name: str`
    * **Derived Property**: `duration: float` (calculated as `end.time - start.time`).
    * **Implementation Notes**:
        * Must have a non-zero, positive duration. A zero-duration event is a `Boundary`.
        * If `name` is not provided, it should default to a string representation of the span (e.g., `"[0.00-15.32]"`).

* **`Segment`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: An ordered sequence of boundaries that partition the span into labeled sections. Represents one layer of annotation.
    * **Attributes**:
        * `boundaries: list[Boundary]` (An ordered list of boundaries defining the segment.)
        * `labels: list[str]` (A list of labels, one for each section created by the boundaries.)
    * **Inherited Attributes**: The `name` attribute from `TimeSpan` serves as the name for the entire annotation layer (e.g., `"functional_harmony"`, `"pop_form"`). `start` and `end` are derived from the first and last boundaries in the list.
    * **Derived Property**: `sections: list[TimeSpan]` (A list of all the labeled time spans that compose the segment.)

* **`MultiSegment`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: The primary input object for analysis, containing multiple `Segment` layers.
    * **Attributes**: `layers: list[Segment]`

* **`BoundaryContour`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: An intermediate, purely structural representation of boundary salience over time.
    * **Attributes**: `boundaries: list[RatedBoundary]`

* **`BoundaryHierarchy`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: The structural output of the monotonic casting process. It contains a well-formed, nested hierarchy of boundaries without semantic labels.
    * **Attributes**: `boundaries: list[LeveledBoundary]`

---
### **Key Methods & Research Pipeline**

The API is designed to support the following workflow for investigating boundary monotonicity.

* **Data Ingestion**:
    * `MultiSegment.from_jams()`
    * `MultiSegment.from_json()`

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