"""
Integration tests for the BNL pipeline.
"""

import json
from pathlib import Path

import pytest

from bnl.core import Hierarchy, ProperHierarchy, RatedBoundaries, RatedBoundary
from bnl.ops import Pipeline
from bnl.strategies import (
    DirectSynthesisStrategy,
    FrequencyStrategy,
    LevelGroupingStrategy,
)


# Define a simple leveling strategy for testing
class BinaryLevelingStrategy(LevelGroupingStrategy):
    def quantize(self, boundaries: RatedBoundaries) -> ProperHierarchy:
        events = boundaries.events
        new_events = [RatedBoundary(time=rb.time, salience=int(rb.salience > 1)) for rb in events]
        # Use the default synthesis strategy on our newly quantized events
        new_boundaries = RatedBoundaries(
            events=new_events,
            start_time=boundaries.start_time,
            end_time=boundaries.end_time,
        )
        return DirectSynthesisStrategy().quantize(new_boundaries)


@pytest.fixture
def estimate_data():
    """Load the MSD-CLASS-CSN-MAGIC estimate file."""
    p = Path(__file__).parent / "fixtures/annotations/8.mp3.msdclasscsnmagic.json"
    with p.open("r") as f:
        return json.load(f)


def test_full_pipeline_from_json_estimate(estimate_data):
    # 1. Load data into core objects
    h = Hierarchy.from_json(estimate_data)

    # 2. Define and wire up the pipeline
    pipeline = Pipeline(
        salience_strategy=FrequencyStrategy(),
        grouping_strategy=None,
        leveling_strategy=BinaryLevelingStrategy(),
    )

    # 3. Process the hierarchy
    result = pipeline.process(h)

    # 4. Assert post-conditions
    assert isinstance(result, ProperHierarchy)
    assert len(result) > 0
    # With this trivial quantizer, we expect two layers (salience 0 and 1)
    assert len(result) == 2
