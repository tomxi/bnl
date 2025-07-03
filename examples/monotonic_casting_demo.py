"""
Demonstration of monotonic casting for bnl.Hierarchy objects.
"""
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for script execution
import matplotlib.pyplot as plt
from bnl import Hierarchy, Segmentation, TimeSpan, ops

def run_demo():
    print("BNL Monotonic Casting Demonstration")
    print("=" * 30)

    # 1. Create a sample non-monotonic Hierarchy
    print("\n1. Creating a sample non-monotonic Hierarchy...")
    # Layer 0: Coarse segmentation
    seg0_boundaries = [0.0, 2.0, 5.0, 7.0]
    seg0_labels = ("A", "B", "C")
    seg0 = Segmentation.from_boundaries(boundaries=seg0_boundaries, labels=seg0_labels, name="Coarse Layer")

    # Layer 1: Finer segmentation, non-monotonic with respect to Layer 0 in some ways
    # e.g., Layer 0 has 2.0, Layer 1 might have 2.5 instead of 2.0
    # Layer 0: (0,2) (2,5) (5,7)
    # Layer 1: (0,1) (1,2.5) (2.5,4) (4,5.5) (5.5,7)
    seg1_boundaries = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0]
    seg1_labels = ("a", "b", "c", "d", "e")
    seg1 = Segmentation.from_boundaries(boundaries=seg1_boundaries, labels=seg1_labels, name="Fine Layer")

    original_hierarchy = Hierarchy(layers=[seg0, seg1], name="Original Non-Monotonic Hierarchy")

    print("\nOriginal Hierarchy:")
    print(repr(original_hierarchy))
    for i, layer in enumerate(original_hierarchy.layers):
        print(f"  Layer {i} ({layer.name}): {layer.boundaries}")

    # 2. Use to_monotonic to convert it
    print("\n2. Converting to a ProperHierarchy using ops.to_monotonic...")
    proper_hierarchy = ops.to_monotonic(original_hierarchy)

    print("\nProper Hierarchy:")
    print(repr(proper_hierarchy))
    for i, layer in enumerate(proper_hierarchy.layers):
        print(f"  Layer {i} ({layer.name}): {layer.boundaries}")

    # 3. Plotting (saving to files)
    print("\n3. Generating plots (will save to files in 'examples' directory)...")

    try:
        # Plot original hierarchy
        fig_orig, ax_orig = original_hierarchy.plot_single_axis(figsize=(10, 4))
        ax_orig.set_title(f"Original: {original_hierarchy.name}")
        plt.tight_layout()
        original_plot_path = "examples/original_hierarchy.png"
        fig_orig.savefig(original_plot_path)
        print(f"   Saved original hierarchy plot to: {original_plot_path}")
        plt.close(fig_orig)

        # Plot proper hierarchy
        fig_prop, ax_prop = proper_hierarchy.plot_single_axis(figsize=(10, 4))
        ax_prop.set_title(f"Proper (Monotonic): {proper_hierarchy.name}")
        plt.tight_layout()
        proper_plot_path = "examples/proper_hierarchy.png"
        fig_prop.savefig(proper_plot_path)
        print(f"   Saved proper hierarchy plot to: {proper_plot_path}")
        plt.close(fig_prop)

    except Exception as e:
        print(f"\nError during plotting: {e}")
        print("Plotting in a script environment might require specific backend configuration.")
        print("This demo is best run in a Jupyter notebook for interactive plots.")

    print("\n" + "=" * 30)
    print("Demo finished.")
    print("To view plots, check the 'examples' directory for .png files.")
    print("You can adapt this script for use in a Jupyter notebook for interactive plotting.")

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    import os
    if not os.path.exists("examples"):
        os.makedirs("examples")
    run_demo()
