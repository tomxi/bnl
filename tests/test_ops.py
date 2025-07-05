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
    SalienceStrategy,
)


@pytest.fixture
def sample_hierarchy():
    """Provides a simple, consistent Hierarchy object for testing."""
    s1 = Segmentation.from_boundaries([0, 2, 4], ["A", "B"])
    s2 = Segmentation.from_boundaries([0, 1, 2, 3, 4], ["a", "b", "c", "d"])
    return Hierarchy([s1, s2], name="TestHierarchy")


# --- Mock Strategies For Testing ---


class PassThroughGrouping(BoundaryGroupingStrategy):
    """A grouping strategy that does nothing."""

    def group(self, boundaries: list[RatedBoundary]) -> list[RatedBoundary]:
        return boundaries


# --- Tests ---


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
    result_hierarchy = pipeline.process(sample_hierarchy)

    # 4. Assert basic post-conditions
    assert result_hierarchy is not None
    assert isinstance(result_hierarchy, ProperHierarchy)
    assert result_hierarchy.name == sample_hierarchy.name
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
    dummy_hierarchy = Hierarchy([Segmentation.from_boundaries([0, 4])])
    ph = pipeline.process(dummy_hierarchy, name="TestPH")

    assert isinstance(ph, ProperHierarchy)
    assert len(ph.layers) == 2  # Salience levels 0 and 1
    assert ph.name == "TestPH"
    assert ph[0].boundaries[1].time == 4.0  # Coarsest has only start/end
    assert ph[1].boundaries[1].time == 2.0  # Finest includes the middle point


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
    dummy_hierarchy = Hierarchy([Segmentation.from_boundaries([0, 4])])
    ph = pipeline.process(dummy_hierarchy)

    assert len(ph) == 1
    # Note: duration is now based on the original hierarchy, not the events
    assert ph.duration == 0.0
    assert len(ph[0].boundaries) == 1  # Should contain just the start time
    assert ph[0].boundaries[0].time == dummy_hierarchy.start.time
