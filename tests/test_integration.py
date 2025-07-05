"""
Integration tests for the BNL pipeline.
"""

import json
from pathlib import Path

import pytest

from bnl.core import MultiSegment, Hierarchy # Old Hierarchy is MultiSegment, Old ProperHierarchy is Hierarchy
from bnl.ops import Pipeline
from bnl.strategies import (
    DirectSynthesisStrategy,
    FrequencyStrategy,
)


@pytest.fixture
def estimate_data():
    """Load the MSD-CLASS-CSN-MAGIC estimate file."""
    p = Path(__file__).parent / "fixtures/annotations/8.mp3.msdclasscsnmagic.json"
    with p.open("r") as f:
        return json.load(f)


def test_full_pipeline_from_json_estimate(estimate_data):
    # 1. Load data into core objects
    ms = MultiSegment.from_json(estimate_data) # Was Hierarchy, now MultiSegment

    # 2. Define and wire up the pipeline
    pipeline = Pipeline(
        salience_strategy=FrequencyStrategy(), # salience_strategy now conceptually rate_strategy
        grouping_strategy=None,
        leveling_strategy=DirectSynthesisStrategy(),
    )

    # 3. Process the multi-segment structure
    result_hierarchy = pipeline(ms) # result is the new Hierarchy (formerly ProperHierarchy)

    # 4. Final validation
    assert isinstance(result_hierarchy, Hierarchy) # Check for new Hierarchy
    assert len(result_hierarchy) > 0 # result.layers

    # The number of layers in the output Hierarchy depends on DirectSynthesisStrategy
    # which creates layers based on unique positive integer rates.
    # FrequencyStrategy calculates rates as counts.

    # Recalculate expected layers based on how DirectSynthesisStrategy works:
    # It uses unique positive integer rates from RatedBoundaries.
    # FrequencyStrategy provides these rates.

    # Get rated boundaries as the pipeline would
    rated_boundaries_from_freq_strategy = FrequencyStrategy().calculate(ms)

    # Filter for positive integer rates, as DirectSynthesisStrategy would use these for levels
    positive_integer_rates = {
        int(rb.rate) for rb in rated_boundaries_from_freq_strategy.events
        if isinstance(rb.rate, (int, float)) and int(rb.rate) > 0
    }

    expected_num_layers = len(positive_integer_rates)

    if not positive_integer_rates and rated_boundaries_from_freq_strategy.events:
        # If all original rates were zero or negative, DirectSynthesisStrategy might create a default single layer
        # or if rates were float and rounded to <1, it might also create a default layer.
        # Based on current DirectSynthesisStrategy, if all rates are invalid (e.g. 0 or negative),
        # it creates a default layer. If events exist but rates are all invalid, it creates one default layer.
        # If no events, it creates one layer.
         expected_num_layers = 1
    elif not rated_boundaries_from_freq_strategy.events :
        expected_num_layers = 1


    assert len(result_hierarchy.layers) == expected_num_layers
