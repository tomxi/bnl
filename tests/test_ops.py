import pytest

from bnl.core import (
    Hierarchy,
    ProperHierarchy,
    RatedBoundaries,
    RatedBoundary,
    Segmentation,
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
def sample_hierarchy() -> Hierarchy:
    """A sample Hierarchy for testing, from coarse to fine."""
    s1 = Segmentation.from_boundaries([0, 4])
    s2 = Segmentation.from_boundaries([0, 2, 4])
    return Hierarchy(start=s1.start, duration=s1.duration, layers=[s1, s2], label="TestHierarchy")


# --- Mock Strategies For Testing ---


class PassThroughGrouping(BoundaryGroupingStrategy):
    """A grouping strategy that does nothing."""

    def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
        return boundaries


# --- Tests ---


def test_coarsest_nonzero_strategy(sample_hierarchy: Hierarchy):
    """Tests the CoarsestNonzeroStrategy for correct salience calculation."""
    strategy = CoarsestNonzeroStrategy()
    rated = strategy.calculate(sample_hierarchy)

    # Expected: num_layers - index
    # Hierarchy has 2 layers.
    # Boundaries at t=0, 4 are in layer 0. Salience = 2 - 0 = 2.
    # Boundary at t=2 is new in layer 1. Salience = 2 - 1 = 1.
    expected_saliences = {0.0: 2, 4.0: 2, 2.0: 1}
    actual_saliences = {rb.time: rb.salience for rb in rated.events}

    assert actual_saliences == expected_saliences


def test_pipeline_runs_without_error(sample_hierarchy):
    """
    Tests that the full pipeline can be constructed and run
    without runtime errors, producing a ProperHierarchy.
    """
    # 1. Setup strategies
    salience_strategy = CoarsestNonzeroStrategy()
    grouping_strategy = PassThroughGrouping()
    leveling_strategy = DirectSynthesisStrategy()

    # 2. Setup the main pipeline
    pipeline = Pipeline(
        salience_strategy=salience_strategy,
        grouping_strategy=grouping_strategy,
        leveling_strategy=leveling_strategy,
    )

    # 3. Process the hierarchy
    result_hierarchy = pipeline(sample_hierarchy)

    # 4. Assert basic post-conditions
    assert result_hierarchy is not None
    assert isinstance(result_hierarchy, ProperHierarchy)
    assert result_hierarchy.label == sample_hierarchy.label
    assert len(result_hierarchy.layers) > 0


def test_pipeline_produces_correct_hierarchy():
    """
    Tests that the pipeline produces a structurally correct ProperHierarchy
    based on a mock set of rated boundaries.
    """

    class MockSalience(SalienceStrategy):
        def calculate(self, hierarchy: Hierarchy) -> RatedBoundaries:
            # This simulates a salience analysis that has already happened.
            events = [
                RatedBoundary(time=0.0, salience=0),
                RatedBoundary(time=2.0, salience=1),
                RatedBoundary(time=4.0, salience=0),
            ]
            return RatedBoundaries(
                events=events,
                start_time=hierarchy.start.time,
                end_time=hierarchy.end.time,
            )

    pipeline = Pipeline(
        salience_strategy=MockSalience(),
        grouping_strategy=None,  # Test the optional case
        leveling_strategy=DirectSynthesisStrategy(),
    )
    # The input hierarchy is a dummy one because the mock strategy ignores it.
    dummy_segmentation = Segmentation.from_boundaries([0, 4])
    dummy_hierarchy = Hierarchy(
        start=dummy_segmentation.start,
        duration=dummy_segmentation.duration,
        layers=[dummy_segmentation],
    )
    ph = pipeline(dummy_hierarchy, label="TestPH")

    assert isinstance(ph, ProperHierarchy)
    assert len(ph.layers) == 2  # Salience levels 0 and 1
    assert ph.label == "TestPH"
    # Layer 0 (salience >= 0) has start, end, and both events
    assert len(ph[0].boundaries) == 3
    # Layer 1 (salience >= 1) has start, end, and the one event
    assert len(ph[1].boundaries) == 3
    assert ph[1].boundaries[1].time == 2.0


def test_pipeline_with_empty_input():
    class MockEmptySalience(SalienceStrategy):
        def calculate(self, hierarchy: Hierarchy) -> RatedBoundaries:
            return RatedBoundaries(
                events=[],
                start_time=hierarchy.start.time,
                end_time=hierarchy.end.time,
            )

    pipeline = Pipeline(
        salience_strategy=MockEmptySalience(),
        grouping_strategy=None,
        leveling_strategy=DirectSynthesisStrategy(),
    )
    dummy_segmentation = Segmentation.from_boundaries([0, 4])
    dummy_hierarchy = Hierarchy(
        start=dummy_segmentation.start,
        duration=dummy_segmentation.duration,
        layers=[dummy_segmentation],
    )
    with pytest.raises(ValueError, match="Hierarchy must contain at least one layer."):
        pipeline(dummy_hierarchy)


def test_rated_boundaries_fluent_api():
    """Tests that the fluent API on RatedBoundaries calls strategies correctly."""

    class MockGrouping(BoundaryGroupingStrategy):
        def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
            # Test that this gets called by adding a boundary with the same salience
            return boundaries + [RatedBoundary(99, 1)]

    class MockLeveling(LevelGroupingStrategy):
        def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
            # Test that this gets called by checking the event count
            return DirectSynthesisStrategy().quantize(boundaries)

    # 1. Initial data
    initial_events = [RatedBoundary(1, 1)]
    rb = RatedBoundaries(events=initial_events, start_time=0.0, end_time=100.0)

    # 2. Chain the calls
    final_ph = rb.group_boundaries(MockGrouping()).quantize_level(MockLeveling())

    # 3. Assertions
    assert isinstance(final_ph, ProperHierarchy)
    # The new event has salience=1, same as the original.
    # Level 0 (salience >= 1) contains start, end, and the two boundaries.
    assert len(final_ph[0].boundaries) == 4
    # There is only one salience level, so only one layer.
    assert len(final_ph.layers) == 1
