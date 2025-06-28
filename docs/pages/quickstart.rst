Quick Start
===========

Installation
------------

.. code-block:: bash

    git clone https://github.com/tomxi/bnl.git
    cd bnl
    pixi install
    pixi run pip install -e .[dev,docs]

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    from bnl import Segmentation, viz

    # Create segmentation from boundaries
    boundaries = [0.0, 2.5, 5.0, 7.5, 10.0]
    labels = ['A', 'B', 'A', 'C']
          seg = Segmentation.from_boundaries(boundaries, labels)

    # Access properties
    print(f"Duration: {seg.end - seg.start}")
    print(f"Labels: {seg.labels}")
    print(f"Boundaries: {seg.bdrys}")

    # Visualize
    fig, ax = plot_segment(seg, text=True)


Loading Data
------------

BNL supports loading real musical structure data from the SALAMI dataset:

.. code-block:: python

    from bnl.data import Dataset

    # Example using the public cloud manifest
    CLOUD_MANIFEST_URL = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/manifest_cloud_boolean.csv"
    dataset = Dataset(CLOUD_MANIFEST_URL)

    # List available track IDs (first 5 for brevity)
    print(f"Found {len(dataset.track_ids)} tracks. First 5: {dataset.track_ids[:5]}")

    # Load a single track (ensure '2' is a valid track_id in the manifest)
    if "2" in dataset.track_ids:
        track = dataset["2"]
        print(f"Loaded: {track}")
        print(f"Track info: {track.info.get('title', 'N/A')}") # Example: get title

        # Load a 'reference' annotation if available
        if "reference" in track.annotations:
            try:
                annotation_obj = track.load_annotation("reference")
                print(f"Loaded 'reference' annotation: {type(annotation_obj)}")
            except Exception as e:
                print(f"Error loading 'reference' annotation for track 2: {e}")
    else:
        print("Track ID '2' not found in dataset.")


.. note::
   For local datasets, you would initialize with a file path, e.g.,
   `dataset = Dataset("~/data/my_local_manifest.csv")`.
   The manifest should point to your data assets.
   Refer to manifest building scripts for conventions.


Next Steps
----------

See :doc:`examples` for more detailed usage examples and :doc:`../api/bnl` for complete API documentation. 