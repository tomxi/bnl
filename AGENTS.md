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
- **Dependencies:** Prefer pixi; minimal external dependencies; keep tomls current and use pixi tasks.
    - **Pixi Tasks:** Use `pixi run check` for a full validation pipeline (format, lint, types, test). Other tasks like `pixi run format`, `pixi run lint`, `pixi run test`, `pixi run docs-build` are also available. Consult `pixi.toml` for details; keep toml files current and run `pixi install` when toml changes.
- **Code Style:** Meaningful naming, logical structure, selective type hints. Adhere to ruff formatting.
- **Data Format:** Prefer feather files; prioritize native core classes over jams/json/mir_eval formats but support them.


## Documentation Conventions

To keep the API reference clean, readable, and easy to maintain, we follow these conventions for docstrings (Google Style):

- **Class Docstrings:** The main class docstring should provide a complete picture.
    - It must include a concise description of the class's purpose.
    - It must have an `Attributes:` section listing all public attributes a user would interact with.
    - For inherited attributes, explicitly list the most relevant ones and note that they are inherited. This makes the class documentation self-contained.
- **`__init__` Docstrings:** The constructor's docstring should be lean.
    - It should only contain an `Args:` section detailing the parameters required for instantiation.
- **Sphinx Configuration:** These conventions work with our Sphinx setup (`conf.py`), which uses `__all__`, an in-docstring `autosummary` directive, and a custom template to control the final output. This fully automated documentation stack provides the cleanest output with the most maintainable source code.


## Agent Notes

*(Add development insights, common patterns, useful commands, etc.)*
- Don't attempt to edit notebooks; feel free to read them, but don't create or edit them.

### User's tasks:
- complete sphinx docs overhaul that's partially succeeding
- already done migrate to google style
- already done getting napolean to pull init description for parameters
- need to respect that I have dataclasses and datafields that I want to show up as parameters, and not attributes.
- I want to be able to scan the page quickly and understand, see parallels, etc. mental mapping
- The methods are still not showing up...
- I need object to show their parent class if they are inhereheted


### User Feedbacks:

### **Key Methods & Research Pipeline**

The API is designed to support the following workflow for investigating boundary monotonicity.

* **Core Transformation Pipeline**:
`MultiSegment -> BoundaryContour -> BoundaryHierarchy ->MultiSegment`
- We'll just use `ops` to operate on the `MultiSegment` directly.


### Lessons from Refactoring `ops.py`

- **Trust the Data Models:** The core data classes in `bnl.core` (e.g., `BoundaryHierarchy`, `Segment`) have validation built-in. For example, a `BoundaryHierarchy` must have at least two boundaries. Functions operating on these objects can and should *assume* they are well-formed. Avoid writing defensive checks for conditions the data model already prevents.
- **Clarity Over Cleverness:** A straightforward, readable implementation (e.g., a simple loop or a generator function) is superior to a complex one-liner that is difficult to debug and understand.
- **Efficient Hierarchy Construction:** When transforming a `BoundaryHierarchy` into a `MultiSegment`, the most efficient strategy is to first group boundaries by level (e.g., into a dictionary) and then iterate through the levels, accumulating boundaries. This is much better than re-filtering the entire list of boundaries for each level.

### Lessons from a Sphinx `autosummary` Debugging Session

A lengthy debugging session to fix the documentation revealed a canonical, maintainable structure for the project's Sphinx docs.

-   **Initial Issue:** `__init__` docstrings were not being correctly integrated into their class documentation. A simple fix (`autoclass_content = "both"`) caused a side effect: a redundant, standalone `__init__` method section appeared on each class page.

-   **Key Insight & Solution:** The root cause was a **nested `toctree` anti-pattern**. The solution was to create a flat structure where a single `api/bnl.rst` file manages the documentation for all modules.

-   **The Canonical Documentation Stack:**
    1.  **Source of Truth:** The `__all__` list in each Python module (`__init__.py`, `core.py`, etc.) defines what gets documented.
    2.  **Single Entry Point:** `docs/api/bnl.rst` uses `autosummary` to generate stub pages for everything in `__all__`.
    3.  **Custom Template:** `docs/_templates/autosummary/class.rst` uses `autoclasstoc` to generate clean class documentation.
    4.  **Clean Configuration:** Simple `conf.py` settings avoid complex hooks.

-   **Final Configuration (docs/conf.py):**
    ```python
    # Use 'class' to prevent dangling transitions
    autoclass_content = "class"
    
    # Configure autoclasstoc for clean class documentation
    autoclasstoc_sections = [
        'public-attrs',      # Shows properties and dataclass fields together
        'public-methods',    # Shows methods without dunder clutter
    ]
    ```

-   **Benefits of This Approach:**
    - ✅ **No "dangling transition" warnings** - eliminated by using `autoclass_content = "class"`
    - ✅ **Clean class documentation** - attributes and parameters grouped together
    - ✅ **No complex hooks needed** - leverages autoclasstoc's built-in capabilities
    - ✅ **Maintainable** - simple configuration that's easy to understand and modify

-   **Key Lesson:** Instead of fighting Sphinx's defaults with complex custom hooks, use the right configuration settings and leverage purpose-built extensions like `autoclasstoc` for clean, maintainable documentation.

### Notebook errors:

```

```
