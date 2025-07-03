"""Tests for operations in bnl.ops."""

import pytest
import numpy as np # For potential array comparisons if needed later
from bnl import Hierarchy, Segmentation, TimeSpan, ProperHierarchy, RatedBoundaries, ops

def test_to_monotonic_simple_conversion():
    """Test conversion of a simple non-monotonic Hierarchy."""
    # Layer 0: [0-2 ("A"), 2-4 ("B")] -> boundaries (0,2,4)
    # Layer 1: [0-1 ("a"), 1-3 ("b"), 3-4 ("c")] -> boundaries (0,1,3,4)
    seg0 = Segmentation.from_boundaries(boundaries=[0.0, 2.0, 4.0], labels=("A", "B"), name="Coarse")
    seg1 = Segmentation.from_boundaries(boundaries=[0.0, 1.0, 3.0, 4.0], labels=("a", "b", "c"), name="Fine")
    non_monotonic_hier = Hierarchy(layers=[seg0, seg1], name="TestHier")

    # Expected boundary depths:
    # 0.0: appears in layer 0 (idx 0) -> depth 0
    # 1.0: appears in layer 1 (idx 1) -> depth 1
    # 2.0: appears in layer 0 (idx 0) -> depth 0
    # 3.0: appears in layer 1 (idx 1) -> depth 1
    # 4.0: appears in layer 0 (idx 0) -> depth 0
    # Sorted rated events for ProperHierarchy:
    # ((0.0,0), (1.0,1), (2.0,0), (3.0,1), (4.0,0))

    proper_hier = ops.to_monotonic(non_monotonic_hier)
    assert isinstance(proper_hier, ProperHierarchy)
    assert proper_hier.name == "TestHier"

    # Max depth is 1, so 2 layers (level_0, level_1)
    assert len(proper_hier.layers) == 2
    assert proper_hier.start == 0.0 and proper_hier.end == 4.0

    # Check Layer 0 of ProperHierarchy (L_idx=0): boundaries from events with d_event <= 0
    # Events: (0.0,0), (2.0,0), (4.0,0)
    # Boundaries: {0.0, 2.0, 4.0}. Global span [0,4] is matched.
    assert proper_hier.layers[0].name == "level_0"
    assert proper_hier.layers[0].boundaries == (0.0, 2.0, 4.0)
    assert proper_hier.layers[0].start == 0.0 and proper_hier.layers[0].end == 4.0

    # Check Layer 1 of ProperHierarchy (L_idx=1): boundaries from events with d_event <= 1
    # Events: (0.0,0), (1.0,1), (2.0,0), (3.0,1), (4.0,0)
    # Boundaries: {0.0, 1.0, 2.0, 3.0, 4.0}. Global span [0,4] is matched.
    assert proper_hier.layers[1].name == "level_1"
    assert proper_hier.layers[1].boundaries == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert proper_hier.layers[1].start == 0.0 and proper_hier.layers[1].end == 4.0

    # Check monotonicity
    assert set(proper_hier.layers[0].boundaries).issubset(set(proper_hier.layers[1].boundaries))

def test_to_monotonic_already_proper():
    """Test with an input Hierarchy that is already effectively monotonic by this rule."""
    # Construct a ProperHierarchy first, then cast its layers to a standard Hierarchy
    events = ((0.0, 0), (1.0, 1), (2.0, 0)) # Max depth 1, 2 layers
    rb = RatedBoundaries(events=events)
    ph_source = ProperHierarchy.from_rated_boundaries(rb, name="AlreadyProperSource")

    # Create a standard Hierarchy from ph_source's layers
    # This standard Hierarchy should be converted back to a similar ProperHierarchy
    standard_hier = Hierarchy(layers=list(ph_source.layers), name="AlreadyProperInput")

    proper_hier_output = ops.to_monotonic(standard_hier)

    assert isinstance(proper_hier_output, ProperHierarchy)
    assert proper_hier_output.name == "AlreadyProperInput"
    assert len(proper_hier_output.layers) == len(ph_source.layers)
    assert proper_hier_output.start == ph_source.start and proper_hier_output.end == ph_source.end

    for i in range(len(ph_source.layers)):
        assert proper_hier_output.layers[i].name == ph_source.layers[i].name # level_i names
        assert proper_hier_output.layers[i].boundaries == ph_source.layers[i].boundaries
        assert proper_hier_output.layers[i].start == ph_source.layers[i].start
        assert proper_hier_output.layers[i].end == ph_source.layers[i].end

def test_to_monotonic_no_internal_boundaries():
    """Test with a Hierarchy where layers have no internal boundaries (single segments)."""
    # Layer 0: [0-5 ("A")] -> boundaries (0,5)
    # Layer 1: [0-5 ("B")] -> boundaries (0,5)
    seg0 = Segmentation.from_boundaries(boundaries=[0.0, 5.0], labels=("A",), name="Coarse")
    seg1 = Segmentation.from_boundaries(boundaries=[0.0, 5.0], labels=("B",), name="Fine")
    hier = Hierarchy(layers=[seg0, seg1], name="NoInternal")

    # Expected boundary depths:
    # 0.0: appears in layer 0 -> depth 0
    # 5.0: appears in layer 0 -> depth 0
    # Sorted rated events: ((0.0,0), (5.0,0))

    proper_hier = ops.to_monotonic(hier)
    assert isinstance(proper_hier, ProperHierarchy)
    assert proper_hier.name == "NoInternal"

    # Max depth 0, so 1 layer (level_0)
    assert len(proper_hier.layers) == 1
    assert proper_hier.start == 0.0 and proper_hier.end == 5.0

    # Layer 0 (L_idx=0): events d_event <= 0. These are (0.0,0), (5.0,0)
    # Boundaries: {0.0, 5.0}
    assert proper_hier.layers[0].name == "level_0"
    assert proper_hier.layers[0].boundaries == (0.0, 5.0)

def test_to_monotonic_empty_boundary_dict():
    """Test when the input hierarchy results in an empty boundary_depths dictionary."""
    seg = Segmentation.from_boundaries(boundaries=[0.0, 1.0], labels=("S1",), name="SingleLayerSeg")
    hier_single_layer_single_segment = Hierarchy(layers=[seg], name="SingleLayerSingleSegment")

    proper_hier = ops.to_monotonic(hier_single_layer_single_segment)
    assert len(proper_hier.layers) == 1
    assert proper_hier.layers[0].boundaries == (0.0, 1.0)
    assert proper_hier.name == "SingleLayerSingleSegment"

def test_to_monotonic_type_error():
    """Test TypeError for incorrect input type."""
    with pytest.raises(TypeError, match="Input must be a bnl.core.Hierarchy object."):
        ops.to_monotonic("not a hierarchy") # type: ignore

def test_boundary_salience_stub(): # Keep the existing stub test for boundary_salience
    """Test the boundary_salience stub function."""
    seg1 = Segmentation.from_boundaries([0.0, 2.0, 5.0], ("A1", "A2"))
    seg2 = Segmentation.from_boundaries([0.0, 1.0, 3.0, 5.0], ("B1", "B2", "B3"))
    dummy_hierarchy = Hierarchy(layers=[seg1, seg2])

    try:
        result = ops.boundary_salience(dummy_hierarchy)
        assert result is None
    except Exception as e:
        pytest.fail(f"ops.boundary_salience raised an exception: {e}")

    try:
        result_with_r = ops.boundary_salience(dummy_hierarchy, r=1.5)
        assert result_with_r is None
    except Exception as e:
        pytest.fail(f"ops.boundary_salience with r param raised an exception: {e}")
