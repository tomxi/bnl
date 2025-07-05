import pytest

from bnl.core import (
    MultiSegment,  # Was Hierarchy
    Hierarchy,     # Was ProperHierarchy
    RatedBoundaries,
    RatedBoundary,
    LeveledBoundary, # For Hierarchy construction
    Segmentation,
    Boundary,      # For basic segmentation
)
from bnl.ops import Pipeline
from bnl.strategies import (
    BoundaryGroupingStrategy,
    CoarsestNonzeroStrategy,
    DirectSynthesisStrategy,
    LevelGroupingStrategy,
    SalienceStrategy,
)


@pytest.fixture
def sample_multisegment() -> MultiSegment: # Renamed from sample_hierarchy
    """A sample MultiSegment for testing, from coarse to fine."""
    s1 = Segmentation.from_boundaries([0, 4]) # These use basic Boundary
    s2 = Segmentation.from_boundaries([0, 2, 4])
    # MultiSegment uses Segmentations with basic Boundary instances
    return MultiSegment(start=s1.start, duration=s1.duration, layers=[s1, s2], label="TestMultiSegment")


# --- Mock Strategies For Testing ---


class PassThroughGrouping(BoundaryGroupingStrategy):
    """A grouping strategy that does nothing."""

    def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
        return boundaries


# --- Tests ---


def test_coarsest_nonzero_strategy(sample_multisegment: MultiSegment): # Use new fixture
    """Tests the CoarsestNonzeroStrategy for correct rate calculation.""" # salience -> rate
    strategy = CoarsestNonzeroStrategy()
    rated = strategy.calculate(sample_multisegment) # Use new fixture name

    # Expected: num_layers - index
    # MultiSegment has 2 layers.
    # Boundaries at t=0, 4 are in layer 0. Rate = 2 - 0 = 2.
    # Boundary at t=2 is new in layer 1. Rate = 2 - 1 = 1.
    expected_rates = {0.0: 2, 4.0: 2, 2.0: 1} # saliences -> rates
    actual_rates = {rb.time: rb.rate for rb in rated.events} # salience -> rate

    assert actual_rates == expected_rates


def test_pipeline_runs_without_error(sample_multisegment: MultiSegment): # Use new fixture
    """
    Tests that the full pipeline can be constructed and run
    without runtime errors, producing a Hierarchy (formerly ProperHierarchy).
    """
    # 1. Setup strategies
    salience_strategy = CoarsestNonzeroStrategy() # Conceptually rate_strategy
    grouping_strategy = PassThroughGrouping()
    leveling_strategy = DirectSynthesisStrategy()

    # 2. Setup the main pipeline
    pipeline = Pipeline(
        salience_strategy=salience_strategy,
        grouping_strategy=grouping_strategy,
        leveling_strategy=leveling_strategy,
    )

    # 3. Process the multi-segment structure
    result_hierarchy = pipeline(sample_multisegment) # Use new fixture name

    # 4. Assert basic post-conditions
    assert result_hierarchy is not None
    assert isinstance(result_hierarchy, Hierarchy) # Check for new Hierarchy class
    assert result_hierarchy.label == sample_multisegment.label
    assert len(result_hierarchy.layers) > 0
    # Check if boundaries are LeveledBoundary
    if len(result_hierarchy.layers) > 0 and len(result_hierarchy.layers[0].boundaries) > 0:
        assert isinstance(result_hierarchy.layers[0].boundaries[0], LeveledBoundary)


def test_pipeline_produces_correct_hierarchy():
    """
    Tests that the pipeline produces a structurally correct Hierarchy (formerly ProperHierarchy)
    based on a mock set of rated boundaries.
    """

    class MockRateStrategy(SalienceStrategy): # Renamed from MockSalience
        def calculate(self, multi_segment: MultiSegment) -> RatedBoundaries: # hierarchy -> multi_segment
            # This simulates a rate analysis that has already happened.
            # Rates must be positive integers for DirectSynthesisStrategy to make LeveledBoundary
            events = [
                RatedBoundary(time=0.0, rate=1), # salience=0 -> rate=1 (must be >0 for LeveledBoundary)
                RatedBoundary(time=2.0, rate=2), # salience=1 -> rate=2
                RatedBoundary(time=4.0, rate=1), # salience=0 -> rate=1
            ]
            return RatedBoundaries(
                events=events,
                start_time=multi_segment.start.time,
                end_time=multi_segment.end.time,
            )

    pipeline = Pipeline(
        salience_strategy=MockRateStrategy(), # Renamed
        grouping_strategy=None,  # Test the optional case
        leveling_strategy=DirectSynthesisStrategy(),
    )
    # The input multi-segment is a dummy one because the mock strategy ignores it.
    # For MultiSegment, use basic Boundary
    dummy_start_boundary = Boundary(0)
    dummy_segmentation = Segmentation(start=dummy_start_boundary, duration=4, boundaries=[dummy_start_boundary, Boundary(4)])
    dummy_multisegment = MultiSegment( # Renamed from dummy_hierarchy
        start=dummy_segmentation.start,
        duration=dummy_segmentation.duration,
        layers=[dummy_segmentation],
    )
    # ph (ProperHierarchy) is now h (Hierarchy)
    h = pipeline(dummy_multisegment, label="TestH") # Renamed ph to h, dummy_hierarchy to dummy_multisegment

    assert isinstance(h, Hierarchy) # Check new Hierarchy
    # Rate levels 1 and 2 will be created by DirectSynthesisStrategy
    # Layer for rate >= 1 will include all events.
    # Layer for rate >= 2 will include event at 2.0.
    # Layers are ordered by rate descending. So layer 0 is for rate 2, layer 1 is for rate 1.
    assert len(h.layers) == 2
    assert h.label == "TestH"

    # Layer 0 (rate >= 2) has start, end, and event at 2.0
    # DirectSynthesisStrategy adds start/end to each layer.
    assert len(h.layers[0].boundaries) == 3
    # Check times: start (0), event (2.0), end (4.0)
    assert h.layers[0].boundaries[0].time == 0.0
    assert h.layers[0].boundaries[1].time == 2.0
    assert h.layers[0].boundaries[2].time == 4.0
    assert isinstance(h.layers[0].boundaries[0], LeveledBoundary) # Check type

    # Layer 1 (rate >= 1) has start, end, and all events (0.0, 2.0, 4.0)
    assert len(h.layers[1].boundaries) == 3
    # Check times: start (0.0), event (2.0), end (4.0)
    # The event at 0.0 and 4.0 had rate 1. Event at 2.0 had rate 2 (>=1).
    # So boundaries should be 0, 2, 4.
    assert h.layers[1].boundaries[0].time == 0.0
    assert h.layers[1].boundaries[1].time == 2.0
    assert h.layers[1].boundaries[2].time == 4.0
    assert isinstance(h.layers[1].boundaries[0], LeveledBoundary)


def test_pipeline_with_empty_input():
    class MockEmptyRate(SalienceStrategy): # Renamed
        def calculate(self, multi_segment: MultiSegment) -> RatedBoundaries: # hierarchy -> multi_segment
            return RatedBoundaries(
                events=[],
                start_time=multi_segment.start.time,
                end_time=multi_segment.end.time,
            )

    pipeline = Pipeline(
        salience_strategy=MockEmptyRate(), # Renamed
        grouping_strategy=None,
        leveling_strategy=DirectSynthesisStrategy(),
    )
    dummy_start_boundary = Boundary(0)
    dummy_segmentation = Segmentation(start=dummy_start_boundary, duration=4, boundaries=[dummy_start_boundary, Boundary(4)])
    dummy_multisegment = MultiSegment( # Renamed
        start=dummy_segmentation.start,
        duration=dummy_segmentation.duration,
        layers=[dummy_segmentation],
    )
    # DirectSynthesisStrategy now handles empty events by creating a default layer.
    # So, it should not raise "Hierarchy must contain at least one layer." from Hierarchy's __post_init__
    # if DirectSynthesisStrategy correctly forms a valid Hierarchy.
    result_h = pipeline(dummy_multisegment)
    assert isinstance(result_h, Hierarchy)
    assert len(result_h.layers) == 1 # Default layer
    assert len(result_h.layers[0].boundaries) == 2 # Start and End LeveledBoundaries
    assert isinstance(result_h.layers[0].boundaries[0], LeveledBoundary)


def test_rated_boundaries_fluent_api():
    """Tests that the fluent API on RatedBoundaries calls strategies correctly."""

    class MockGrouping(BoundaryGroupingStrategy):
        def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
            # Test that this gets called by adding a boundary
            # Ensure rate is positive for LeveledBoundary
            return boundaries + [RatedBoundary(99, rate=1)] # salience -> rate

    class MockLeveling(LevelGroupingStrategy):
        def quantize(self, boundaries: RatedBoundaries) -> Hierarchy: # ProperHierarchy -> Hierarchy
            # Test that this gets called by checking the event count
            return DirectSynthesisStrategy().quantize(boundaries)

    # 1. Initial data
    initial_events = [RatedBoundary(1, rate=1)] # salience -> rate
    rb = RatedBoundaries(events=initial_events, start_time=0.0, end_time=100.0)

    # 2. Chain the calls
    final_h = rb.group_boundaries(MockGrouping()).quantize_level(MockLeveling()) # final_ph -> final_h

    # 3. Assertions
    assert isinstance(final_h, Hierarchy) # ProperHierarchy -> Hierarchy
    # The new event has rate=1, same as the original.
    # DirectSynthesisStrategy makes layers from unique positive rates.
    # All events (original and mocked) + start/end will have level 1.
    # So, one layer for level 1.
    assert len(final_h.layers) == 1
    # Boundaries in that layer: start, event at 1, event at 99, end.
    assert len(final_h.layers[0].boundaries) == 4
    assert isinstance(final_h.layers[0].boundaries[0], LeveledBoundary)
