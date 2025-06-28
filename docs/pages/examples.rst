Examples
========

Music Structure Analysis
-------------------------

.. code-block:: python

    import numpy as np
    from bnl import Segmentation, viz

    # Define song structure boundaries
    boundaries = [0.0, 15.2, 45.8, 78.3, 92.1, 120.0]
    labels = ['intro', 'verse', 'chorus', 'verse', 'outro']
    song = Segmentation.from_boundaries(boundaries, labels)


Visualization
-------------

.. code-block:: python

    # Basic plot
    fig, ax = viz.plot_segment(song, text=True, ytick="Song Structure")
    
    # Compare segmentations
    import matplotlib.pyplot as plt
    
    reference = Segmentation.from_boundaries([0.0, 15.0, 45.0, 78.0, 92.0, 120.0], 
                                             ['intro', 'verse', 'chorus', 'verse', 'outro'])
    prediction = Segmentation.from_boundaries([0.0, 16.5, 44.2, 80.1, 120.0], 
                                              ['intro', 'verse', 'chorus', 'outro'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    viz.plot_segment(reference, ax=axes[0], text=True, ytick="Reference")
    viz.plot_segment(prediction, ax=axes[1], text=True, ytick="Prediction")
    plt.tight_layout()


Working with SALAMI Data
-------------------------

Loading and exploring the SALAMI dataset:

.. code-block:: python

    from bnl.data import Dataset
    from bnl.core import Hierarchy, Segmentation # Assuming these might be returned
    import matplotlib.pyplot as plt

    # Assuming a manifest file exists, e.g., for a local SALAMI subset:
    # LOCAL_MANIFEST_PATH = "~/data/salami_subset/metadata.csv"
    # Or for the cloud dataset:
    CLOUD_MANIFEST_URL = "https://pub-05e404c031184ec4bbf69b0c2321b98e.r2.dev/manifest_cloud_boolean.csv"

    # Initialize the Dataset
    # For local: dataset = Dataset(LOCAL_MANIFEST_PATH)
    dataset = Dataset(CLOUD_MANIFEST_URL)
    print(f"Available SALAMI tracks: {len(dataset.track_ids)}")
    
    # Load a specific track by its ID
    # Ensure the track_id exists in your manifest
    if "10" in dataset.track_ids:
        track = dataset["10"]
        print(f"Track: {track}")

        # Access metadata
        info = track.info
        print(f"Artist: {info.get('artist', 'N/A')}") # Use .get for safety
        print(f"Title: {info.get('title', 'N/A')}")
        if 'duration' in info:
            print(f"Duration: {info['duration']:.1f}s")

        # Load a specific annotation, e.g., the 'reference' JAMS annotation
        # The key 'reference' comes from how the manifest was built
        # (e.g. has_annotation_reference -> 'reference')
        if "reference" in track.annotations:
            try:
                annotation_data = track.load_annotation("reference")
                print(f"Loaded annotation type: {type(annotation_data)}")
                if isinstance(annotation_data, Hierarchy):
                    print(f"  Hierarchy with {len(annotation_data.layers)} layers.")
                elif isinstance(annotation_data, Segmentation):
                    print(f"  Segmentation with {len(annotation_data.segments)} segments.")
            except Exception as e:
                print(f"Could not load 'reference' annotation for track 10: {e}")
        else:
            print("Track 10 does not have a 'reference' annotation listed.")

    # Load multiple tracks for analysis
    sample_track_ids = ["10", "100", "1000"] # Ensure these are in your dataset
    valid_sample_tracks = []
    for tid in sample_track_ids:
        if tid in dataset.track_ids:
            valid_sample_tracks.append(dataset[tid])
        else:
            print(f"Track ID {tid} not found in dataset.")

    if valid_sample_tracks:
        durations = [t.info.get('duration', 0.0) for t in valid_sample_tracks]
        print(f"Sample durations for existing tracks: {durations}")

.. note::
   For local datasets, ensure your manifest file correctly points to your data assets
   (e.g., JAMS annotations, audio files) relative to the manifest's location or using
   conventions understood by the path reconstruction logic in `bnl.data.Dataset`.
   The example manifest builder scripts (`scripts/build_local_manifest.py`)
   expect specific directory structures (like `jams/` and `audio/` subfolders).
