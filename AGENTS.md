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
    - **Pixi Tasks:** Use `pixi run check` for a full validation pipeline (format, lint, types, test). Other tasks like `pixi run format`, `pixi run lint`, `pixi run test`, `pixi run docs-build` are also available. Consult `pixi.toml` for details.
- **Code Style:** Meaningful naming, logical structure, selective type hints. Adhere to ruff formatting.
- **Data Format:** Prefer feather files; prioritize native core classes over jams/json/mir_eval formats but support them.

## Key Changes & Deprecations
- **`bnl.metrics` Module Removed:** The internal `bnl.metrics` module has been removed. For established evaluation metrics, use `frameless-eval`. For frame-based metrics and other utilities, `mir_eval` (already a dependency) should be used. This decision simplifies the codebase and relies on well-maintained external libraries.

## Agent Notes

*(Add development insights, common patterns, useful commands, etc.)*

### User is working on...
- Streamlining and stabilizing the core data classes and API.
- Improving test coverage using new fixtures.
- Ensuring documentation is accurate and helpful.

### User Feedbacks
- "put in a cell in the development,ipynb to make it super easy to go in and start developing and debugging." (Ongoing consideration)
- "make better choices around wha'ts a property and what's a classmethod etc. Read the generated docs to think through what's doing what." (Addressed in recent `bnl.core` refactoring)
- "We don't need the metrics modules anymore. We will use frameless-eval for doing established metrics, and use mir_eval for frame based metrics, and other utilities." (Implemented: `bnl.metrics` removed)
- "API design: Ideally, track.load_hierarchy should be track.load_annotation(anno_type)"
    - THIS IS NOW IMPLEMENTED: `Track.load_annotation(annotation_type, annotation_id=None)` handles JAMS (multi_segment, other segmentations, specific selection) and JSON hierarchies. (This note seems current and correct).

### Concrete Examples for `Track.load_annotation`
(This section seems current and useful, retaining as is)
- Some concrete examples:
local_track.annotations
{'adobe-mu1gamma1': PosixPath('/Users/tomxi/data/salami/adobe/def_mu_0.1_gamma_0.1/2.mp3.msdclasscsnmagic.json'),
 'adobe-mu1gamma9': PosixPath('/Users/tomxi/data/salami/adobe/def_mu_0.1_gamma_0.9/2.mp3.msdclasscsnmagic.json'),
 'adobe-mu5gamma5': PosixPath('/Users/tomxi/data/salami/adobe/def_mu_0.5_gamma_0.5/2.mp3.msdclasscsnmagic.json'),
 'reference': PosixPath('/Users/tomxi/data/salami/jams/2.jams')}

cloud_track.annotations
{'adobe-mu1gamma1': 'https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/adobe/def_mu_0.1_gamma_0.1/2.mp3.msdclasscsnmagic.json',
 'adobe-mu1gamma9': 'https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/adobe/def_mu_0.1_gamma_0.9/2.mp3.msdclasscsnmagic.json',
 'adobe-mu5gamma5': 'https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/adobe/def_mu_0.5_gamma_0.5/2.mp3.msdclasscsnmagic.json',
 'reference': 'https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/jams/2.jams'}

So I want: track.load_hierarchy(anno_type) to return a Hierarchy object, with anno_type being track.annotations' keys.
