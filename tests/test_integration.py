"""
Integration tests for the BNL pipeline.
"""

import json
from pathlib import Path

import pytest

from bnl.core import Hierarchy, ProperHierarchy
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
    h = Hierarchy.from_json(estimate_data)

    # 2. Define and wire up the pipeline
    pipeline = Pipeline(
        salience_strategy=FrequencyStrategy(),
        grouping_strategy=None,
        leveling_strategy=DirectSynthesisStrategy(),
    )

    # 3. Process the hierarchy
    result = pipeline(h)

    # 4. Final validation
    assert isinstance(result, ProperHierarchy)
    assert len(result) > 0
    # The number of layers should correspond to the number of unique
    # frequency counts in the hierarchy's boundaries.
    all_boundaries = [b.time for layer in h for b in layer.boundaries]
    unique_freqs = len(set(all_boundaries.count(b) for b in set(all_boundaries)))
    assert len(result) == unique_freqs
