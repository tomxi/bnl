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
- **Documentation:** Clear docstrings for complex logic and public APIs; Sphinx + RTD ready, google style docstrings; build and consult the docs actively during development, planning, and ideation
- **Dependencies:** Managed with `uv` and `virtualenv` using `pyproject.toml`.
    - **Setup:** `uv venv` to create a virtual environment, then `source .venv/bin/activate`.
    - **Installation:** Install core and optional dependencies (`dev`, `docs`, `dash`, `notebooks`) with `uv pip install -e '.[dev,docs,dash,notebooks]'`.
    - **Updates:** Re-run the installation command to update dependencies.
- **Running Commands:** Use `uv run <command>` to execute scripts defined in `pyproject.toml` or any command within the virtual environment without explicit activation (e.g., `uv run python -m pytest`).
- **Best Practice:** `pyproject.toml` is the single source of truth for all dependencies (core, dev, docs, etc.). `uv` and `virtualenv` provide a fast and isolated environment for development.
- **Code Style:** Meaningful naming, logical structure, selective type hints.

## Agent Notes

*(Add development insights, common patterns, useful commands, etc.)*

### User's tasks:
- we just commited to plotly, now let's clean the data module
- we want to peel off the app eventually too, and clean the dependencies.
- Next steps: make sure we have track.ests and track.refs working for both local and remote tracks
- Optional: better manifest file? i.e. save the manifestfile in the repo.
- Solidify the chainable logic and strategy in ops.py. We want to expose the strategies classes too. Two ways for me to use it. If I want to inspect the innards of the strategy I can do that by keeping the object around. The chainable method will wrap around the object and return the final result only.
- let's the the non-bmeasure eval metrics working on the Dash app.
  

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

