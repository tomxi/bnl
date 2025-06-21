#!/usr/bin/env python3
"""
Experiment runner for the "Lose the Frame" paper.
"""

from pqdm.processes import pqdm
from bnl import fio, mtr, H
import sys
import os
import time
import xarray as xr
import mir_eval


def time_track_depth_sweep(tid, cache_dir="./depth_sweep", retime=False):
    """Time depth sweep experiment for a single track."""
    # Check if already timed
    os.makedirs(cache_dir, exist_ok=True)
    output_filepath = os.path.join(cache_dir, f"{tid}.nc")
    if os.path.exists(output_filepath) and not retime:
        print(f"Already timed {tid}.")
        return output_filepath

    adobe_hiers = fio.salami_adobe_hiers(tid=str(tid))
    ref_hiers = fio.salami_ref_hiers(tid=str(tid))
    
    # Get the first hierarchy from each dictionary
    adobe_hier = list(adobe_hiers.values())[0]
    salami_hier = list(ref_hiers.values())[0]
    
    # Extract intervals and labels from hierarchy objects
    ref_itvls, ref_labels, est_itvls, est_labels = mtr.align_hier(
        salami_hier.itvls, salami_hier.labels, 
        adobe_hier.itvls, adobe_hier.labels
    )
    
    # Create hierarchy objects from aligned data
    ref = H(ref_itvls, ref_labels)
    est = H(est_itvls, est_labels)
    
    # Create results array
    result_da = xr.DataArray(
        dims=["level", "tid", "output", "frame_size"],
        coords={
            "level": range(est.d),
            "tid": [tid],
            "output": ["run_time", "lp", "lr", "lf"],
            "frame_size": [0, 0.1, 0.2, 0.5, 1, 2],
        },
    )

    # Run timing experiments
    for d in range(est.d):
        for frame_size in result_da.coords["frame_size"]:
            start_time = time.time()
            if frame_size == 0:
                lp, lr, lm = mtr.lmeasure(
                    ref.itvls, ref.labels, est.itvls[: d + 1], est.labels[: d + 1]
                )
            else:
                lp, lr, lm = mir_eval.hierarchy.lmeasure(
                    ref.itvls,
                    ref.labels,
                    est.itvls[: d + 1],
                    est.labels[: d + 1],
                    frame_size=frame_size,
                )
            run_time = time.time() - start_time

            result_da.loc[dict(level=d, tid=tid, output="run_time", frame_size=frame_size)] = run_time
            result_da.loc[dict(level=d, tid=tid, output="lp", frame_size=frame_size)] = lp
            result_da.loc[dict(level=d, tid=tid, output="lr", frame_size=frame_size)] = lr
            result_da.loc[dict(level=d, tid=tid, output="lf", frame_size=frame_size)] = lm

    # Save results
    result_da.to_netcdf(output_filepath)
    print(f"Timed {tid} and saved to {output_filepath}.")
    return output_filepath


def run_depth_sweep_experiment():
    """Run the time depth sweep experiment."""
    # Filter tracks that have both Adobe hierarchies and 2 reference hierarchies
    valid_tids = [
        tid for tid in fio.salami_tids() 
        if fio.salami_adobe_hiers(tid=str(tid)) and len(fio.salami_ref_hiers(tid=str(tid))) == 2
    ]
    
    print(f"Found {len(valid_tids)} valid tracks for depth sweep experiment")
    
    if not valid_tids:
        print("No valid tracks found. Experiment cannot run.")
        return
    
    # Suppress warnings in worker processes
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Run experiment in parallel
    results = pqdm(
        valid_tids,
        time_track_depth_sweep,
        n_jobs=8,
        desc="Running depth sweep experiments"
    )
    
    # Report results
    successful_files = [r for r in results if r and r != 0]
    print(f"Completed: {len(successful_files)}/{len(valid_tids)} files created")
    if len(successful_files) < len(valid_tids):
        print(f"Warning: {len(valid_tids) - len(successful_files)} experiments failed")


def main():
    """Main function for command-line usage."""
    exp_methods = {
        "depth_sweep": run_depth_sweep_experiment,
        "all": run_depth_sweep_experiment,
    }

    if len(sys.argv) == 1:
        command = "all"
    elif len(sys.argv) == 2:
        command = sys.argv[1]
    else:
        print("Lose the Frame Experiment Runner")
        print("Usage: python lose_the_frame_exps.py [command]")
        print("Commands: depth_sweep, all")
        sys.exit(1)

    if command in exp_methods:
        exp_methods[command]()
        print(f"✓ Completed: {command}")
    else:
        print(f"✗ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
