# Agent Instructions

Concise guidance for AI agents working with this music information retrieval codebase.

## Core Principles

- **Simplicity First:** Keep code elegant, concise, and maintainable. Avoid over-engineering. Cleanliness is godliness.
- **Core Classes:** Built around classes in `bnl/core/` - use extensively and keep robust
- **Thoughtful API Design:** Design the API for the core classes to be intuitive and easy to use; consider the user's perspective; this is tooling for researchers, not for developers;
- **Research Focus:** Support music structure analysis and representation research questions
- **80/20 Rule:** Achieve maximum impact with minimal effort

## Development Guidelines

- **Testing:** Maintain tests pragmatically without slowing development; make tests failure driven;
- **Documentation:** Clear docstrings for complex logic and public APIs; Sphinx + RTD ready, google style docstrings; build and consult the docs actively during development, planning, and ideation
- **Code Style:** Meaningful naming, logical structure, selective type hints.
- **Dependencies:** Managed locally with `conda`. use the `py311` conda environment.
## Agent Notes

*(Add development insights, common patterns, useful commands, etc.)*

### User's tasks:
- we want to peel off the app eventually too, and clean the dependencies.
- Solidify the chainable logic and strategy in ops.py. We want to expose the strategies classes too. Two ways for me to use it. If I want to inspect the innards of the strategy I can do that by keeping the object around. The chainable method will wrap around the object and return the final result only.
- let's get the T-measure eval metrics working on the jupyter notebook.

### User Feedbacks:

### **Key Methods & Research Pipeline**

The API is designed to support the following workflow for investigating boundary monotonicity.

* **Core Transformation Pipeline**:
`MultiSegment -> BoundaryContour -> BoundaryHierarchy ->MultiSegment`
- We have a chainable command pattern for this.


## Lessons Learnt (ARCHIVE):
### Lessons from Refactoring `ops.py`

- **Trust the Data Models:** The core data classes in `bnl.core` (e.g., `BoundaryHierarchy`, `Segment`) have validation built-in. For example, a `BoundaryHierarchy` must have at least two boundaries. Functions operating on these objects can and should *assume* they are well-formed. Avoid writing defensive checks for conditions the data model already prevents.
- **Clarity Over Cleverness:** A straightforward, readable implementation (e.g., a simple loop or a generator function) is superior to a complex one-liner that is difficult to debug and understand.
- **Efficient Hierarchy Construction:** When transforming a `BoundaryHierarchy` into a `MultiSegment`, the most efficient strategy is to first group boundaries by level (e.g., into a dictionary) and then iterate through the levels, accumulating boundaries. This is much better than re-filtering the entire list of boundaries for each level.

