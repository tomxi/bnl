import pytest

from bnl import core

# Test data
contour = core.BoundaryContour(
    bs=[
        core.RatedBoundary(0, 1.0),
        core.RatedBoundary(1, 0.5),
        core.RatedBoundary(2, 1.0),
        core.RatedBoundary(4, 1.0),  # gap here
        core.RatedBoundary(5, 1.0),
    ]
)


def test_clean_boundaries_absorb():
    """Tests the 'absorb' strategy for clean_boundaries."""
    cleaned = contour.clean(strategy="absorb")
    assert len(cleaned.bs) == 3
    assert cleaned.bs[1].time == 2.0


def test_clean_boundaries_invalid_strategy():
    """Tests that an invalid strategy raises a ValueError."""
    with pytest.raises(ValueError):
        contour.clean(strategy="invalid_strategy")


def test_level_by_distinct_salience():
    """Tests the level_by_distinct_salience function."""
    contour_for_leveling = core.BoundaryContour(
        bs=[
            core.RatedBoundary(0, 10.0),
            core.RatedBoundary(1, 20.0),
            core.RatedBoundary(2, 10.0),
            core.RatedBoundary(3, 30.0),
        ]
    )
    hierarchy = contour_for_leveling.level(strategy="unique")
    assert isinstance(hierarchy, core.BoundaryHierarchy)
    assert len(hierarchy.bs) == 4
    # Check levels: 10.0 -> 1, 20.0 -> 2, 30.0 -> 3
    # Expected levels: 10.0 -> level 1, 20.0 -> level 2, 30.0 -> level 3
    # The start and end boundaries get the max level (3)
    # The order of boundaries is not guaranteed, so we check by time.
    # The salience of beginning and end are ignored and added back at the end
    expected_levels = {0.0: 2, 1.0: 2, 2.0: 1, 3.0: 2}
    found_levels = {b.time: b.level for b in hierarchy.bs}
    assert found_levels == expected_levels


def test_boundary_salience_by_count():
    """Test salience calculation with 'count' strategy."""
    s1 = core.Segment.from_itvls([[0, 1], [1, 5]], ["A", "B"])
    s2 = core.Segment.from_itvls([[0, 2], [2, 5]], ["a", "b"])
    s3 = core.Segment.from_itvls([[0, 1], [1, 5]], ["x", "y"])
    ms = core.MultiSegment(raw_layers=[s1, s2, s3], name="test_ms")

    hierarchy = ms.contour(strategy="count")
    assert isinstance(hierarchy, core.BoundaryHierarchy)
    assert hierarchy.name == "test_ms"

    # Times: 0, 1, 2, 5
    # Counts: 0 appears 3 times, 1 appears 2 times, 2 appears 1 time, 5 appears 3 times
    expected_levels = {0.0: 3, 1.0: 2, 2.0: 1, 5.0: 3}
    assert len(hierarchy.bs) == len(expected_levels)

    for b in hierarchy.bs:
        assert b.level == expected_levels[b.time]


def test_boundary_salience_invalid_strategy():
    """Test that an invalid salience strategy raises a ValueError."""
    s1 = core.Segment.from_bs([0, 1], ["A"], name="L1")
    ms = core.MultiSegment(raw_layers=[s1], name="test_ms")
    with pytest.raises(ValueError):
        ms.contour(strategy="invalid_strategy")


def test_clean_boundaries_kde():
    """Tests the 'kde' strategy for clean_boundaries."""
    contour = core.BoundaryContour(
        bs=[
            core.RatedBoundary(0, 1.0),
            core.RatedBoundary(1, 0.5),
            core.RatedBoundary(1.1, 0.5),
            core.RatedBoundary(2, 0.8),
            core.RatedBoundary(2.2, 0.8),
            core.RatedBoundary(3, 1.0),
        ]
    )
    cleaned = contour.clean(strategy="kde")
    assert isinstance(cleaned, core.BoundaryContour)
    assert len(cleaned.bs) < len(contour.bs)


def test_boundary_salience_by_depth():
    """Test salience calculation with 'depth' strategy."""
    s1 = core.Segment.from_itvls([[0, 1], [1, 5]], ["A", "B"])  # coarsest, salience=3
    s2 = core.Segment.from_itvls([[0, 2], [2, 5]], ["a", "b"])  # medium, salience=2
    s3 = core.Segment.from_itvls([[0, 1], [1, 5]], ["x", "y"])  # finest, salience=1
    ms = core.MultiSegment(raw_layers=[s1, s2, s3], name="test_ms")

    hierarchy = ms.contour(strategy="depth")
    assert isinstance(hierarchy, core.BoundaryHierarchy)
    assert hierarchy.name == "test_ms"

    # Boundaries at t=0, 1, 5 are in s1 (salience=3).
    # Boundary at t=2 is only in s2 (salience=2).
    # The salience is determined by the coarsest layer (reversed enumerate).
    # reversed -> s3(1), s2(2), s1(3)
    # t=0: in s3,s2,s1 -> gets salience 3 from s1
    # t=1: in s3,s1 -> gets salience 3 from s1
    # t=2: in s2 -> gets salience 2 from s2
    # t=5: in s3,s2,s1 -> gets salience 3 from s1
    expected_levels = {0.0: 3, 1.0: 3, 2.0: 2, 5.0: 3}
    assert len(hierarchy.bs) == len(expected_levels)
    for b in hierarchy.bs:
        assert b.level == expected_levels[b.time]


def test_boundary_salience_by_prob():
    """Test salience calculation with 'prob' strategy."""
    # Layer 1: 1 effective boundary, weight = 1/1 = 1
    s1 = core.Segment.from_itvls([[0, 1], [1, 5]], ["A", "B"])
    # Layer 2: 1 eff bdr, weight = 1/1 = 1
    s2 = core.Segment.from_itvls([[0, 2], [2, 5]], ["a", "b"])
    # Layer 3: 3 eff bdr, weight = 1/3 = 0.33
    s3 = core.Segment.from_itvls([[0, 1], [1, 2], [2, 4], [4, 5]], ["w", "x", "y", "z"])
    ms = core.MultiSegment(raw_layers=[s1, s2, s3], name="test_ms")

    contour = ms.contour(strategy="prob")
    assert isinstance(contour, core.BoundaryContour)
    assert contour.name == "test_ms"

    # Boundaries:
    # t=0: s1(1) + s2(1) + s3(0.33) = 2.33
    # t=1: s1(1) + s3(0.33) = 1.33
    # t=2: s2(1) + s3(0.33) = 1.33
    # t=4: s3(0.33) = 0.33
    # t=5: s1(1) + s2(1) + s3(0.33) = 2.33
    expected_saliences = {0.0: 7 / 3, 1.0: 4 / 3, 2.0: 4 / 3, 4.0: 1 / 3, 5.0: 7 / 3}
    assert len(contour.bs) == len(expected_saliences)

    # Use pytest.approx for float comparison
    from pytest import approx

    found_saliences = {b.time: b.salience for b in contour.bs}
    for time, salience in expected_saliences.items():
        assert found_saliences[time] == approx(salience)
