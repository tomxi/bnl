from . import fio, fmtr
import xarray as xr
import os, time, mir_eval, warnings
import numpy as np


# warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")


def time_salami_track(tid, out_dir="./new_compare/"):
    hiers = fio.salami_ref_hiers(tid)
    if len(hiers) < 2:
        # print(f"Track {tid} has only one hierarchies, skipping.")
        return

    # Check if the output directory exists, if not create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Build fname
    fname = os.path.join(out_dir, f"{tid}.nc")
    if os.path.exists(fname):
        # print(f"Track {tid} already exists, just load..")
        return

    da_coords = dict(
        frame_size=[0, 0.1, 0.2, 0.5, 1, 2],
        output=["run_time", "p", "r", "f"],
        metric=["lmeasure", "pairwise", "vmeasure"],
    )
    # Create a dataarray for this track's results
    result_da = xr.DataArray(dims=da_coords.keys(), coords=da_coords)

    # Get the two hierarchies
    ref = hiers[0]
    est = hiers[1]

    for fs in da_coords["frame_size"]:
        out = time_metric(ref, est, frame_size=fs)
        # return result_da, run_time, scores
        for m in da_coords["metric"]:
            options = dict(frame_size=fs, metric=m)
            result_da.loc[options] = out[m]

    # save the results
    result_da.to_netcdf(fname)
    return fname


def time_metric(ref, est, frame_size=0):
    if frame_size == 0:
        start_time = time.time()
        lme = fmtr.lmeasure(ref.itvls, ref.labels, est.itvls, est.labels)
        lme_time = time.time() - start_time

        start_time = time.time()
        pfc = fmtr.pairwise(
            ref.itvls[-1], ref.labels[-1], est.itvls[-1], est.labels[-1]
        )
        pfc_time = time.time() - start_time

        start_time = time.time()
        vme = fmtr.vmeasure(
            ref.itvls[-1], ref.labels[-1], est.itvls[-1], est.labels[-1]
        )
        vme_time = time.time() - start_time
    else:
        start_time = time.time()
        lme = mir_eval.hierarchy.lmeasure(
            ref.itvls, ref.labels, est.itvls, est.labels, frame_size=frame_size
        )
        lme_time = time.time() - start_time
        start_time = time.time()
        pfc = mir_eval.segment.pairwise(
            ref.itvls[-1],
            ref.labels[-1],
            est.itvls[-1],
            est.labels[-1],
            frame_size=frame_size,
        )
        pfc_time = time.time() - start_time
        start_time = time.time()
        vme = mir_eval.segment.vmeasure(
            ref.itvls[-1],
            ref.labels[-1],
            est.itvls[-1],
            est.labels[-1],
            frame_size=frame_size,
        )
        vme_time = time.time() - start_time
    return dict(
        lmeasure=[lme_time, *lme], pairwise=[pfc_time, *pfc], vmeasure=[vme_time, *vme]
    )
