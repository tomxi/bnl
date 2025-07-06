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

### **Core API Handoff Summary (Decoupled Labels)**

**Guiding Principle**: The design separates pure structural markers (`Boundary`) from their descriptive metadata (`label`). Labels are managed by the container objects in which the boundaries frame.

#### **I. Point-like Objects 📍**

These objects are now pure, label-less structural markers.

* **`Boundary`**
    * **Inherits From**: None (Base Class)
    * **Purpose**: A raw, unannotated marker on a timeline.
    * **Attributes**: `time: float`

* **`RatedBoundary`**
    * **Inherits From**: `Boundary`
    * **Purpose**: A boundary with a continuous measure of importance.
    * **Attributes**: `salience: float`

* **`LeveledBoundary`**
    * **Inherits From**: `RatedBoundary`
    * **Purpose**: A definitive structural node within a `Hierarchy`, identified only by its time and level.
    * **Attributes**: `level: positive integer`
    * **Note**: Its inherited `salience` should be set to its integer `level`.

---

#### **II. Span-like Objects (Containers) 🌊**

These objects now manage the labels for the boundaries they contain.

* **`TimeSpan`**
    * **Inherits From**: None (Base Class)
    * **Purpose**: Represents a generic time interval.
    * **Attributes**: `start: Boundary`, `end: Boundary`, `label: str`
    * **Derived Property**: `duration: float` (The difference between `start.time` and `end.time`).
    * **Note**: Strictly enforced to have a non-zero positive duration. Zero duration is a Boundary. Label has to not be None, if not provided, it should be something like a string representation of the Span [start.time - end.time].

* **`Segment`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: An ordered sequence of boundaries with associated labels.
    * **Attributes**:
        * `boundaries: list[Boundary]`
        * `labels: list[str]`
    * **Derived Property**: `sections: list[TimeSpan]` (A list of all the time spans in the segment).
    * **Note**: super's start and end are the first and last boundaries in the list, label is 

* **`BoundaryContour`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: An intermediate representation of salience over time. It is pure and contains no label information.
    * **Attributes**: `boundaries: list[RatedBoundary]`

* **`BoundaryHierarchy`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: A boundary hierarchy with wellformed layers of boundaries, still needs labeling for each layer.
    * **Attributes**: `boundaries: list[LeveledBoundary]`

* **`MultiSegment`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: The primary input for analysis, containing multiple `Segment` layers.
    * **Attributes**: `layers: list[Segment]`

* **`Hierarchy`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: Montonoic Hierarchical Boundaries, still needs labeling for each layer.
    * **Attributes**:
        * `boundaries: BoundaryHierarchy`
        * `labels: list[list[str]]`
    * **Derived Property**: `layers: list[Segment]` (A list of all the layers of the hierarchy). This makes the Hierarchy a MultiSegment, so run super init in the constructor for inheritance.

---
### **Object key methods**
* Strictly require labels to be not None. If it's None, use a string representation of the Span [start.time - end.time].
* We need `from_json` and `from_jams` methods for `MultiSegment`.
* We need plotting methods for `MultiSegment.plot()`: this will be our main plotting method.
    * Build the plotting method for `TimeSpan` thoughtfully if appropriate and capable to make the API cleaner and more contained.
* main pipeline for doing monotonic boundary casting which is the key research work right now:
    * The main flow for boundary monotonic casting is `MultiSegment` (no labels) -> `BoundaryContour` -> `BoundaryHierarchy` + (inject labels) -> `Hierarchy`. We are not dealing with labels and just focusing on the Boundary.
    * `MultiSegment.to_contour()` -> `BoundaryContour`.
    * `BoundaryContour.to_levels()` -> `BoundaryHierarchy`.
    * `Hierarchy.to_multisegment()` -> `MultiSegment`.

---
### **Plotting Architecture 🎨**

Plotting is handled hierarchically. Container objects orchestrate the plotting of their components by passing down a shared styling context. The public API remains a simple `.plot()` method on container objects.

* **`TimeSpan.plot()`**: The fundamental drawing method. It plots a single labeled, colored span (`axvspan`) based on the styles it receives.
* **`Segment.plot()`**: Composes a plot by calling `.plot()` on each of its internal `TimeSpan` sections.
* **`MultiSegment.plot()` & `Hierarchy.plot()`**: These are the top-level entry points. They are responsible for creating the styling context and orchestrating the entire plot.

#### **The `PlottingStyleMaps` Mechanism**

1.  **Context Creation**: Before drawing, the top-level `.plot()` call scans all components to pre-build a `PlottingStyleMaps` object. This object maps every unique label to a consistent style (e.g., `'Verse'` → `'C0'`).
2.  **Context Passing**: This `PlottingStyleMaps` object is passed down as an internal argument through the entire call stack (`MultiSegment` → `Segment` → `TimeSpan`).
3.  **Coordinated Drawing**: Each `TimeSpan.plot()` method uses the received style map to look up the correct style for its label, ensuring a globally consistent plot.