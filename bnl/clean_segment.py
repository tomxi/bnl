import numpy as np


def print_verbose(text, verbose):
    """Print text if verbose

    Parameters
    ----------
    text : string
        string to print
    verbose : bool
        If True print text
    """
    if verbose:
        print(text)


def clean_segments(levels, min_duration=8, fix_level=3, verbose=False):
    """
    Given segmentation levels (in the format returned by reindex(), take segmentation level fix_level (note that levels
    start at 1, not 0) and merge out all segments shorter than min_duration. Returns the cleaned up segments as a 2D
    numpy array.

    Cleaning of short segments follows this pseudo algorithm:

    PSEUDOALG:
    while shortest_seg < min_duaration and n_segs > 1:
        if current_seg = start of song
            merge with next_seg
        elif current_seg = end of song
            merge with previous_seg
        elif id_prev_seg == id_next_seg
            merge prev + current + next
        else:
            while not solved and level is not lowest:
                look down 1 level, get id of segment with most overlap
                if lower id is same as prev or next ID, use it, we're done.
                else, repeat
                ADDITION: if we go all the way down to level 1 and still dont
                find a good ID, check which of the short segment's boundaries
                overlaps the most with boundaries at lower levels and keep that
                one (and merge short seg with the seg adjacent to the losing
                boundary).

    Parameters
    ----------
    levels: list
        List of segmentations [[itvl1, lbl1], [itvl2, lbl2], ...] as returned by reindex()
    min_duration: float
        Minimum duration to keep a segment
    fix_level:
        Segmentation level to be fixed (note: starts at 1)
    verbose:
        Verbose printouts if true.

    Returns
    -------
    segs: np.ndarray
        Cleaned segments from level fix_level as 2D numpy array
    """

    def get_segs(levs, level=3):
        """
        Utility function to get segments for a specific segmentation level in
        numpy array format. Specify the level start at 1.

        Parameters
        ----------
        levs: segmentation levels as returned by reindex()
        level: level of segmentation to return (minimum is 1).

        Returns
        -------

        """
        ts, ids = levs[level - 1][0], levs[level - 1][1]
        array = []
        for (start, end), i in zip(ts, ids):
            array.append([start, end, int(i)])

        return np.asarray(array)

    def durations(segs):
        """
        Given segments in the format returned by get_segs(), return an array
        with the durations of the segments

        Parameters
        ----------
        segs: np.ndarray
            Segments array in the format returned by get_segs()

        Returns
        -------
        durations: np.ndarray
            Array of segment durations

        """
        return segs[:, 1] - segs[:, 0]

    def merge_segs(segs, first_idx, last_idx, new_id):
        """
        Given segments in the format returned by get_segs(), i.e. a 2D numpy
        array, the indices if the first and last segments to merge (indexing
        starts at 0), and the id for the new merged segment, merge the segments
        and return a new segments array.


        Parameters
        ----------
        segs: np.ndarray
            Segments in the format returned by get_segs()
        first_idx: int
            Index of first segment to merge
        last_idx: int
            Index of last segment to merge
        new_id: int
            id for the merged segment

        Returns
        -------
        segs: np.ndarray
            Merged segments in 2D numpy array format
        """
        assert first_idx < last_idx
        new_segs = []

        new_start = None
        new_end = None

        for n, seg in enumerate(segs):
            if n < first_idx:
                new_segs.append(seg)
            elif n == first_idx:
                new_start = seg[0]
            elif n < last_idx:
                pass
            elif n == last_idx:
                new_end = seg[1]
                new_segs.append(np.asarray([new_start, new_end, new_id]))
            else:
                new_segs.append(seg)

        return np.asarray(new_segs)

    def get_overlap_time(s1, s2):
        """
        Get the overlap duration (in seconds) between segments s1 and s2. If the
        segments don't overlap at all a duration of 0 is returned.

        Params:
        s1/s2 = list or tuple of the form (start_time, end_time)

        Parameters
        ----------
        s1: list/np.ndarray/tuple
        s2: list/np.ndarray/tuple

        Returns
        -------
        overlap: float
            Overlap duration between s1 and s2 in seconds.
        """

        max_start = max(s1[0], s2[0])
        min_end = min(s1[1], s2[1])
        return max(min_end - max_start, 0)

    def get_down_id(minidx, segs, dsegs):
        """
        Given the segments segs at a certain segmentation level, segments at
        a lower segmentation level dsegs, and the index minidx of the shortest
        segment in segs, return the segment ID of the segment in dsegs that
        overlaps the most with the segment given by segs[minidx].

        Parameters
        ----------
        minidx: int
            Index of the segment in segs (starts at 0)
        segs: np.ndarray
            Segments in 2D numpy array format
        dsegs: np.ndarray
            Segments at a lower level than segs, in 2D numpy array format

        Returns
        -------
        downid: int
            ID of the segment in dsegs that overlaps the most (in time) with
            the segment given by segs[minidx].
        """
        seg_times = segs[minidx, :2]
        down_times = dsegs[:, :2]
        overlaps = []
        for dt in down_times:
            overlaps.append(get_overlap_time(seg_times, dt))

        max_overlap_idx = np.argmax(overlaps)
        downid = dsegs[max_overlap_idx, 2]

        return downid

    def get_boundary_overlap(levels, fix_level, minidx, max_distance=1, verbose=False):
        """
        Given a segment specified by its fix_level and minidx, count how many
        segments at lower levels have a start time that overlaps with the
        segment's start and end times.

        Parameters
        ----------
        levels: list
            Segmeentations as returned by reindex()
        fix_level: int
            Level of segmentation to consider (starts at 1)
        minidx: int
            Index of the short segment whose boundaries will be examined
        max_distance: float
            The max distance (in seconds) between two boundaries to consider
            them as overlapping.

        Returns
        -------
        boundary_overlap_start: int
            Count of boundaries at lower levels that overlap with the segment's
            start time
        boundary_overlap_end: int
            Count of boundaries at lower levels that overlap with the segment's
            end time.
        """
        segs = get_segs(levels, level=fix_level)
        # print_verbose("---------> Looking for boundary overlap for minidx: {}, start: {:.1f}, end: {:.1f}".format(
        #     minidx, segs[minidx, 0], segs[minidx, 1]), verbose)

        boundary_overlap_start = 0
        boundary_overlap_end = 0

        downlevel = fix_level - 1
        while downlevel > 0:
            dsegs = get_segs(levels, downlevel)
            for ds in dsegs:
                if np.abs(ds[0] - segs[minidx, 0]) <= max_distance:
                    # print_verbose("---------> start boundary ({:.1f}) +1 at downlevel {} at time {:.1f}".format(
                    #     segs[minidx, 0], downlevel, ds[0]), verbose)
                    boundary_overlap_start += 1
                if np.abs(ds[0] - segs[minidx, 1]) <= max_distance:
                    # print_verbose("---------> end boundary ({:.1f}) +1 at downlevel {} at time {:.1f}".format(
                    #     segs[minidx, 1], downlevel, ds[0]), verbose)
                    boundary_overlap_end += 1
            downlevel -= 1

        return boundary_overlap_start, boundary_overlap_end

    # ********** BEGINNING OF CLEAN_SEGMENTS **********
    if fix_level <= 1:
        return get_segs(levels, level=1)

    segs = get_segs(levels, level=fix_level)
    downid = None

    id_to_fix = np.max(segs[:, 2])

    # OLD STRATEGY: always fix first the shortest segment. Could lead to sob-optimal results in rare cases.
    # while min(durations(segs)) < min_duration and len(segs) > 1:  # repeat until no short segs left or just 1 seg left

    # NEW STRATEGY: first fix the small segments of the highest seg ID, then move to previous seg ID, etc., till
    # We reach seg ID 0 which is the last seg ID to fix. This gives short segments of lower IDs priority over short
    # segs of higher IDs.
    while id_to_fix >= 0 and len(segs) > 1:

        # NEW SAMPLING STRATEGY:
        if not np.any(segs[:, 2] == id_to_fix):
            id_to_fix -= 1
            continue

        rows_with_id = segs[segs[:, 2] == id_to_fix]
        id_durations = durations(rows_with_id)
        minrow = rows_with_id[np.argmin(id_durations)]
        minidx = None
        for i in range(len(segs)):
            if np.allclose(segs[i], minrow):
                minidx = i
                break

        if durations(segs)[minidx] > min_duration:
            id_to_fix -= 1
            continue

        # OLD SAMPLING STRATEGY:
        # minidx = np.argmin(durations(segs))  # find shortest seg

        downlevel = fix_level - 1  # must be inside loop so it resets at each iteration!

        if minidx == 0:  # if first seg, merge with next seg
            print_verbose(
                "merging first (level: {}, minidx: {})".format(fix_level, minidx),
                verbose,
            )
            new_id = segs[minidx + 1, 2]
            segs = merge_segs(segs, 0, 1, new_id)

        elif minidx == len(segs) - 1:  # if last seg, merge with prev seg
            print_verbose(
                "merging last (level: {}, minidx: {})".format(fix_level, minidx),
                verbose,
            )
            new_id = segs[minidx - 1, 2]
            segs = merge_segs(segs, minidx - 1, minidx, new_id)

        elif (
            segs[minidx - 1, 2] == segs[minidx + 1, 2]
        ):  # if seg ID same as prev and next, merge the 3
            print_verbose(
                "merging same before/after (level: {}, minidx: {})".format(
                    fix_level, minidx
                ),
                verbose,
            )
            new_id = segs[minidx - 1, 2]
            segs = merge_segs(segs, minidx - 1, minidx + 1, new_id)
        else:  # otherwise consult lower level for seg ID
            # GO DOWN A LEVEL FOR CONSULT
            print_verbose(
                "consluting lower levels (this level: {}, lower level: {}, minidx: {}, id: {})".format(
                    fix_level, downlevel, minidx, segs[minidx, 2]
                ),
                verbose,
            )
            solved = False
            while not solved:
                dsegs = get_segs(levels, downlevel)
                # find majority seg ID overlapping with current seg
                downid = get_down_id(minidx, segs, dsegs)
                # get neighboring IDs
                neighborids = [segs[minidx - 1, 2], segs[minidx + 1, 2]]

                print_verbose(
                    "---> this level: {}, lower level: {}, id: {}, downid: {})".format(
                        fix_level, downlevel, segs[minidx, 2], downid
                    ),
                    verbose,
                )

                if (
                    downid not in neighborids and downlevel > 1
                ):  # if lower level is same ID and not 0, go down
                    downlevel -= 1
                else:  # otherwise use seg ID from lower lev and merge
                    # USE downid
                    if segs[minidx - 1, 2] == downid == segs[minidx + 1, 2]:
                        print_verbose(
                            "------> found good downid: {}, merging (prev/next)".format(
                                downid
                            ),
                            verbose,
                        )
                        segs = merge_segs(segs, minidx - 1, minidx + 1, downid)
                    elif downid == segs[minidx - 1, 2]:
                        print_verbose(
                            "------> found good downid: {}, merging (prev)".format(
                                downid
                            ),
                            verbose,
                        )
                        segs = merge_segs(segs, minidx - 1, minidx, downid)
                    elif downid == segs[minidx + 1, 2]:
                        print_verbose(
                            "------> found good downid: {}, merging (next)".format(
                                downid
                            ),
                            verbose,
                        )
                        segs = merge_segs(segs, minidx, minidx + 1, downid)
                    else:
                        print_verbose(
                            "------> NO good downid, looking at boundary overlap",
                            verbose,
                        )
                        # In this case we could NOT find a good downid and have gone all the way down to level 1
                        # So, instead, we look at the short segment's start/end times, and see which one overlaps the
                        # most with start/end times at lower levels. Whichever overlaps the most is kept as a boundary
                        # and the other is merged with its adjacent section and take on its ID.
                        print_verbose("minidx: {}".format(minidx), verbose)
                        print_verbose("segs:\n{}".format(segs), verbose)
                        boundary_overlap_start, boundary_overlap_end = (
                            get_boundary_overlap(
                                levels,
                                fix_level,
                                minidx,
                                max_distance=1,
                                verbose=verbose,
                            )
                        )
                        if boundary_overlap_start > boundary_overlap_end:
                            print_verbose(
                                "---------> Start wins ({}/{}), merging next".format(
                                    boundary_overlap_start, boundary_overlap_end
                                ),
                                verbose,
                            )
                            segs = merge_segs(
                                segs, minidx, minidx + 1, segs[minidx + 1, 2]
                            )
                        else:
                            print_verbose(
                                "---------> End wins ({}/{}), merging prev".format(
                                    boundary_overlap_start, boundary_overlap_end
                                ),
                                verbose,
                            )
                            segs = merge_segs(
                                segs, minidx - 1, minidx, segs[minidx - 1, 2]
                            )
                        print_verbose("new segs:\n{}".format(segs), verbose)
                    solved = True

    # print_verbose("Done cleaning segments.", verbose)
    return segs
