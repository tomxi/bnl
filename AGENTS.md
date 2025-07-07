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
- Building up the plotting API very cleanly.

### User Feedbacks
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

#### **The `StyleMaps` Mechanism**

1.  **Context Creation**: Before drawing, `MultiSegment.plot()` scans all components to pre-build a `StyleMaps` object. This object maps every unique label to a consistent style.
2.  **Context Passing**: This `StyleMaps` object is passed down as an internal argument through the call stack (`MultiSegment` → `Segment` → `TimeSpan`).
3.  **Coordinated Drawing**: Each `TimeSpan.plot()` method uses the received style map to look up the correct style for its label, ensuring a globally consistent and readable plot.


old style map generation:
```python
def label_style_dict(labels, boundary_color="white", **kwargs):
    """
    Creates a mapping of labels to visual style properties for consistent visualization.

    This function processes a list of labels (which may contain nested lists),
    extracts unique labels, sorts them, and assigns consistent visual styles
    (like colors and hatch patterns) to each label.

    Parameters
    ----------
    labels : nparray or list
        List of labels (can contain nested lists). Duplicate labels will be handled only once.
    **kwargs : dict
        Additional style properties to apply to all labels. These will override
        the default styles if there are conflicts.

    Returns
    -------
    dict
        A dictionary mapping each unique label to its style properties.
        Each entry contains properties like 'facecolor', 'edgecolor', 'linewidth',
        'hatch', and 'label'.

    Notes
    -----
    - If there are 80 or fewer unique labels, uses 'tab10' colormap with 8 hatch patterns.
    - If there are more than 80 unique labels, uses 'tab20' colormap with 15 hatch patterns.
    - Default styles include white edgecolor and linewidth of 1.
    """
    # Find unique elements in labels. Labels can be list of arrays, list of labels, or a single array
    unique_labels = np.unique(
        np.concatenate([np.atleast_1d(np.asarray(l)) for l in labels])
    )
    # This modification ensures that even a single label is treated as a 1-dimensional array.

    # More hatch patterns for more labels
    hatchs = ["", "..", "O.", "*", "xx", "xxO", "\\O", "oo", "\\"]
    more_hatchs = [h + "--" for h in hatchs]

    if len(unique_labels) <= 80:
        hatch_cycler = cycler(hatch=hatchs)
        fc_cycler = cycler(color=plt.get_cmap("tab10").colors)
        p_cycler = hatch_cycler * fc_cycler
    else:
        hatch_cycler = cycler(hatch=hatchs + more_hatchs)
        fc_cycler = cycler(color=plt.get_cmap("tab20").colors)
        # make it repeat...
        p_cycler = itertools.cycle(hatch_cycler * fc_cycler)

    # Create a mapping of labels to styles by cycling through the properties
    # and assigning them to the labels as they appear in the unique labels' ordering
    seg_map = dict()
    for lab, properties in zip(unique_labels, p_cycler):
        # set style according to p_cycler
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }
        # Swap color -> facecolor here so we preserve edgecolor on rects
        if "color" in style:
            style.setdefault("facecolor", style["color"])
            style.pop("color", None)
        seg_map[lab] = dict(linewidth=1, edgecolor=boundary_color)
        seg_map[lab].update(style)
        seg_map[lab].update(kwargs)
        seg_map[lab]["label"] = lab
    return seg_map


def segment(
    intervals,
    labels,
    ax,
    text=False,
    ytick="",
    time_ticks=False,
    style_map=None,
):
    """Plot a single layer of flat segmentation."""
    ax.set_xlim(intervals[0][0], intervals[-1][-1])

    if style_map is None:
        style_map = label_style_dict(labels, edgecolor="white")
    transform = ax.get_xaxis_transform()

    for ival, lab in zip(intervals, labels):
        rect = ax.axvspan(ival[0], ival[1], ymin=0, ymax=1, **style_map[lab])
        if text:
            ann = ax.annotate(
                lab,
                xy=(ival[0], 1),
                xycoords=transform,
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                clip_on=True,
                bbox=dict(boxstyle="round", facecolor="white"),
            )
            ann.set_clip_path(rect)

    if time_ticks:

        # Use the default automatic tick locator
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(TimeFormatter())
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xticks([])

    if ytick == "":
        ax.set_yticks([])
    else:
        ax.set_yticks([0.5])
        ax.set_yticklabels([ytick])
    return ax.get_figure(), ax
```