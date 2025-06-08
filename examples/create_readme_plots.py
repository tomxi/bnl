import numpy as np
import matplotlib.pyplot as plt # Import for saving plots
from bnl.core import Segmentation

def create_and_save_plots():
    """
    Creates example flat and hierarchical segmentations,
    plots them, and saves the plots to the examples/images/ directory.
    """
    # 1. Create and plot a flat segmentation
    flat_itvls = np.array([[0, 5], [5, 10], [10, 15]])
    flat_labels = ['A', 'B', 'A']
    # Note: Creating a Segmentation object for a flat segmentation
    # is_hierarchical=False is inferred if itvls is a single array and not a list of arrays.
    flat_seg = Segmentation(flat_itvls, flat_labels, sr=20) # sr is needed for Bhat, and plot might use it for ticks

    fig_flat, ax_flat = flat_seg.plot()
    fig_flat.savefig("examples/images/flat_segmentation_example.png", bbox_inches='tight')
    plt.close(fig_flat) # Close the figure to free memory

    # 2. Create and plot a hierarchical segmentation
    hier_itvls = [
        np.array([[0, 15]]),
        np.array([[0, 5], [5, 10], [10, 15]])
    ]
    hier_labels = [
        ['X'],
        ['A', 'B', 'A']
    ]
    # is_hierarchical=True is inferred if itvls is a list of arrays.
    hier_seg = Segmentation(hier_itvls, hier_labels, sr=20)

    fig_hier, axs_hier = hier_seg.plot(legend=True) # Example with legend
    fig_hier.savefig("examples/images/hierarchical_segmentation_example.png", bbox_inches='tight')
    plt.close(fig_hier)

    print("Plots generated and saved to examples/images/")

if __name__ == "__main__":
    create_and_save_plots()
