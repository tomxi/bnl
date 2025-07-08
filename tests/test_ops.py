from bnl import core, ops


def test_boundary_salience_by_count():
    """Test salience calculation with 'count' strategy."""
    s1 = core.Segment.from_itvls([[0, 1], [1, 5]], ["A", "B"])
    s2 = core.Segment.from_itvls([[0, 2], [2, 5]], ["a", "b"])
    s3 = core.Segment.from_itvls([[0, 1], [1, 5]], ["x", "y"])
    ms = core.MultiSegment([s1, s2, s3], name="test_ms")

    hierarchy = ops.boundary_salience(ms, strategy="count")
    assert isinstance(hierarchy, core.BoundaryHierarchy)
    assert hierarchy.name == "test_ms"

    # Times: 0, 1, 2, 5
    # Counts: 0 appears 3 times, 1 appears 2 times, 2 appears 1 time, 5 appears 3 times
    expected_levels = {0.0: 3, 1.0: 2, 2.0: 1, 5.0: 3}
    assert len(hierarchy.boundaries) == len(expected_levels)

    for b in hierarchy.boundaries:
        assert b.level == expected_levels[b.time]


def test_boundary_salience_by_depth():
    """Test salience calculation with 'depth' strategy."""
    s1 = core.Segment.from_itvls([[0, 1], [1, 5]], ["A", "B"])  # coarsest, salience=3
    s2 = core.Segment.from_itvls([[0, 2], [2, 5]], ["a", "b"])  # medium, salience=2
    s3 = core.Segment.from_itvls([[0, 1], [1, 5]], ["x", "y"])  # finest, salience=1
    ms = core.MultiSegment([s1, s2, s3], name="test_ms")

    hierarchy = ops.boundary_salience(ms, strategy="depth")
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
    assert len(hierarchy.boundaries) == len(expected_levels)
    for b in hierarchy.boundaries:
        assert b.level == expected_levels[b.time]


def test_level_by_distinct_salience():
    """Test level_by_distinct_salience function."""
    boundaries = [
        core.RatedBoundary(0.0, 10.0),
        core.RatedBoundary(1.0, 2.5),
        core.RatedBoundary(2.0, 5.0),
        core.RatedBoundary(3.0, 2.5),
        core.RatedBoundary(4.0, 7.5),
        core.RatedBoundary(5.0, 10.0),
    ]
    bc = core.BoundaryContour("test_bc", boundaries)

    hierarchy = ops.level_by_distinct_salience(bc)
    assert isinstance(hierarchy, core.BoundaryHierarchy)
    assert hierarchy.name == "test_bc"

    # Unique saliences (inner): 2.5, 5.0, 7.5
    # Ranks (levels): 2.5 -> 1, 5.0 -> 2, 7.5 -> 3
    # Max level for outer boundaries is 3.
    # Expected:
    # t=0 -> level 3 (outer)
    # t=1 -> sal 2.5 -> level 1
    # t=2 -> sal 5.0 -> level 2
    # t=3 -> sal 2.5 -> level 1
    # t=4 -> sal 7.5 -> level 3
    # t=5 -> level 3 (outer)
    expected_levels = {
        0.0: 3,
        1.0: 1,
        2.0: 2,
        3.0: 1,
        4.0: 3,
        5.0: 3,
    }

    # The order is not guaranteed by the function, so we check by time
    found_boundaries = {b.time: b for b in hierarchy.boundaries}
    assert len(found_boundaries) == len(expected_levels)
    for time, level in expected_levels.items():
        assert found_boundaries[time].level == level


def test_boundary_salience_by_prob():
    """Test salience calculation with 'prob' strategy."""
    # Layer 1: 1 effective boundary, weight = 1/1 = 1
    s1 = core.Segment.from_itvls([[0, 1], [1, 5]], ["A", "B"])
    # Layer 2: 1 eff bdr, weight = 1/1 = 1
    s2 = core.Segment.from_itvls([[0, 2], [2, 5]], ["a", "b"])
    # Layer 3: 3 eff bdr, weight = 1/3 = 0.33
    s3 = core.Segment.from_itvls([[0, 1], [1, 2], [2, 4], [4, 5]], ["w", "x", "y", "z"])
    ms = core.MultiSegment([s1, s2, s3], name="test_ms")

    contour = ops.boundary_salience(ms, strategy="prob")
    assert isinstance(contour, core.BoundaryContour)
    assert contour.name == "test_ms"

    # Boundaries:
    # t=0: s1(1) + s2(1) + s3(0.33) = 2.33
    # t=1: s1(1) + s3(0.33) = 1.33
    # t=2: s2(1) + s3(0.33) = 1.33
    # t=4: s3(0.33) = 0.33
    # t=5: s1(1) + s2(1) + s3(0.33) = 2.33
    expected_saliences = {0.0: 7 / 3, 1.0: 4 / 3, 2.0: 4 / 3, 4.0: 1 / 3, 5.0: 7 / 3}
    assert len(contour.boundaries) == len(expected_saliences)

    # Use pytest.approx for float comparison
    from pytest import approx

    found_saliences = {b.time: b.salience for b in contour.boundaries}
    for time, salience in expected_saliences.items():
        assert found_saliences[time] == approx(salience)
