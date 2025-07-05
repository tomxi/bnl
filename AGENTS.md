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
- getting the notebook working
- getting the app working
- moving on to getting the montoonic casting working

### User Feedbacks
- the from_jams and from_json methods are still not working... we need to fix them.
- hmm I need a way to quary the label at a timepoint, or the closest boundary in past time basically.

### Recent Progress
- ✅ **Implemented `naive_salience` strategy** in `ops.py` that follows the specification from the Core API Handoff Summary
- ✅ **Created `SaliencePayload` structure** that returns both `BoundaryContour` and `LabelContextMap` as specified  
- ✅ **Updated `MultiSegment.to_contour()`** to work with the new payload structure
- ✅ **Fixed test naming** for `to_contour` method
- ✅ **Verified implementation** with comprehensive label context collection from all layers

### **Core API Handoff Summary**

**Guiding Principle**: The design separates simple, single-purpose data objects from the complex analysis functions that create them. Transformation context is passed alongside objects, not inside them.

#### **I. Point-like Objects 📍**

These classes represent single moments in time.

* **`Boundary`**
    * **Inherits From**: None (Base Class)
    * **Purpose**: A basic, labeled marker on a timeline.
    * **Attributes**: `time: float`, `label: str | None`

* **`RatedBoundary`**
    * **Inherits From**: `Boundary`
    * **Purpose**: A boundary with a continuous measure of importance (salience).
    * **Attributes**: `salience: float`

* **`LeveledBoundary`**
    * **Inherits From**: `RatedBoundary`
    * **Purpose**: A definitive node within a `Hierarchy`.
    * **Attributes**: `ancestry: list[str]` (An ordered list of labels from the root to the current level).
    * **Derived Property**: `level: int` (Returns `len(self.ancestry)`).
    * **Note**: Its inherited `salience` should be set to its integer `level`. Its inherited `label` should be set to the last item in its `ancestry`.

---

#### **II. Span-like Objects (Containers) 🌊**

These classes represent spans of time and collections of point-like objects.

* **`TimeSpan`**
    * **Inherits From**: None (Base Class)
    * **Purpose**: Represents a generic time interval.
    * **Attributes**: `start: Boundary`, `duration: float`

* **`Segment`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: An ordered sequence of simple boundaries representing a single layer.
    * **Attributes**: `boundaries: list[Boundary]`

* **`MultiSegment`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: A container for multiple `Segment` layers; the primary input for analysis.
    * **Attributes**: `layers: list[Segment]`

* **`BoundaryContour`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: An intermediate representation of salience over time.
    * **Attributes**: `boundaries: list[RatedBoundary]`

* **`Hierarchy`**
    * **Inherits From**: `TimeSpan`
    * **Purpose**: The final, structured output of the analysis.
    * **Attributes**: `boundaries: list[LeveledBoundary]`
    * **Key Method**: `to_multisegment() -> MultiSegment`

---

### **Core Analysis Workflow**

The transformation from input to output is a two-stage process that uses a temporary, transient data structure to pass context.

1.  **Analyze Input**: An analysis function takes a **`MultiSegment`**. It produces a payload containing two items:
    * A **`BoundaryContour`** object representing salience.
    * A **`LabelContextMap`** (`dict[float, list[str]]`), which maps each boundary's time to its original list of labels from all layers.

2.  **Build Hierarchy**: A second function takes this entire payload as input. It uses the `BoundaryContour` to identify significant points and the `LabelContextMap` to look up the corresponding labels needed to construct the `ancestry` for each **`LeveledBoundary`**. It then returns the final **`Hierarchy`**.

