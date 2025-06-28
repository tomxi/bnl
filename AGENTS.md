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
- **Dependencies:** Prefer pixi; minimal external dependencies; keep tomls current and use pixi tasks
- **Code Style:** Meaningful naming, logical structure, selective type hints
- **Data Format:** Prefer feather files; prioritize native core classes over jams/json/mir_eval formats but support them;

## Agent Notes

*(Add development insights, common patterns, useful commands, etc.)*

### User is working on...
- Now I'm working to fix load_hierarchy for non-jams files. also the api for getting annotations is not good...
- Now let's check the core dataclasses creation methods.
- hmmm but that requires me to load the json or jams file first, which the current api is a messy mess.. I'm stopping and thinking about the api design.

### User Feedbacks
- put in a cell in the development,ipynb to make it super easy to go in and start developing and debugging.
- make better choices around wha'ts a property and what's a classmethod etc. Read the generated docs to think through what's doing what.
- We don't need the mertics modules anymore. We will use frameless-eval for doing established metrics, and use mir_eval for frame based metrics, and other utilities.
- API design: Soideall, track.load_hierarchy should be track.load_annotation(anno_type)

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
