from . import fio, mtr
import xarray as xr
import os, time, mir_eval, warnings, json
import numpy as np


# warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")


def time_salami_track(tid, out_dir="./new_faster_compare/"):
    hiers = fio.salami_ref_hiers(tid)
    if len(hiers) < 2:
        # print(f"Track {tid} has only one hierarchies, skipping.")
        return

    ref, est = hiers.values()
    # Build fname
    fname = os.path.join(out_dir, f"{tid}.nc")
    if os.path.exists(fname):
        # print(f"Track {tid} already exists, just load..")
        return

    # Check if the output directory exists, if not create it
    os.makedirs(out_dir, exist_ok=True)

    da_coords = dict(
        tid=[tid],
        frame_size=[0, 0.1, 0.2, 0.5, 1, 2],
        output=["run_time", "p", "r", "f"],
        metric=["lmeasure", "pairwise", "vmeasure"],
    )
    # Create a dataarray for this track's results
    result_da = xr.DataArray(dims=da_coords.keys(), coords=da_coords)

    # Get the two hierarchies

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
        lme = mtr.lmeasure(ref.itvls, ref.labels, est.itvls, est.labels)
        lme_time = time.time() - start_time

        start_time = time.time()
        pfc = mtr.pairwise(ref.itvls[-1], ref.labels[-1], est.itvls[-1], est.labels[-1])
        pfc_time = time.time() - start_time

        start_time = time.time()
        vme = mtr.vmeasure(ref.itvls[-1], ref.labels[-1], est.itvls[-1], est.labels[-1])
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


def time_depth_sweep(tid, frame_size=0.2, cache_dir="./depth_sweep", retime=False):

    # Check if already timed
    os.makedirs(cache_dir, exist_ok=True)
    output_filepath = os.path.join(cache_dir, f"{tid}.nc")
    if os.path.exists(output_filepath) and not retime:
        print(f"Already timed {tid}.")
        return output_filepath

    adobe_hier = fio.adobe_hiers(tid=str(tid))
    salami_hier = fio.salami_ref_hiers(tid=str(tid))[0]
    ref, est = mtr.align_hier(salami_hier, adobe_hier)
    # Save the results to xarray
    result_da = xr.DataArray(
        dims=["level", "tid", "version", "output"],
        coords={
            "level": range(est.d),
            "tid": [tid],
            "version": ["mir_eval", "my"],
            "output": ["run_time", "lp", "lr", "lf"],
        },
    )

    for d in range(est.d):
        start_time = time.time()
        mylp, mylr, mylm = fmtr.lmeasure(
            ref.itvls, ref.labels, est.itvls[: d + 1], est.labels[: d + 1]
        )
        my_run_time = time.time() - start_time
        result_da.loc[dict(level=d, tid=tid, version="my")] = [
            my_run_time,
            mylp,
            mylr,
            mylm,
        ]

        start_time = time.time()
        melp, melr, melm = mir_eval.hierarchy.lmeasure(
            ref.itvls,
            ref.labels,
            est.itvls[: d + 1],
            est.labels[: d + 1],
            frame_size=frame_size,
        )
        me_run_time = time.time() - start_time
        result_da.loc[dict(level=d, tid=tid, version="mir_eval")] = [
            me_run_time,
            melp,
            melr,
            melm,
        ]

    # Save the results to a NetCDF file
    result_da.to_netcdf(output_filepath)
    print(f"Timed {tid} and saved to {output_filepath}.")
    return output_filepath


def time_single_anno(tid, frame_size=0.1, cache_dir="./single_anno", retime=False):
    salami_hiers = list(fio.salami_ref_hiers(tid=str(tid)).values())
    if len(salami_hiers) > 1:
        # print(f"Track {tid} has multiple hierarchies, skipping.")
        return
    salami_hier = salami_hiers[0]

    # Check if already timed
    os.makedirs(cache_dir, exist_ok=True)
    output_filepath = os.path.join(cache_dir, f"{tid}.nc")
    if os.path.exists(output_filepath) and not retime:
        print(f"Already timed {tid}.")
        return output_filepath

    adobe_hier = list(fio.salami_adobe_hiers(tid=str(tid)).values())[0]
    ref_itvls, ref_labels, est_itvls, est_labels = mtr.align_hier(
        salami_hier.itvls, salami_hier.labels, adobe_hier.itvls, adobe_hier.labels
    )

    # Build the xarray for results
    result_da = xr.DataArray(
        dims=["tid", "version"],
        coords={
            "tid": [str(tid)],
            "version": ["mir_eval", "my"],
        },
    )

    start_time = time.time()
    mtr.lmeasure(ref_itvls, ref_labels, est_itvls, est_labels)
    my_run_time = time.time() - start_time
    result_da.loc[dict(tid=tid, version="my")] = my_run_time

    start_time = time.time()
    mir_eval.hierarchy.lmeasure(
        ref_itvls, ref_labels, est_itvls, est_labels, frame_size=frame_size
    )
    me_run_time = time.time() - start_time
    result_da.loc[dict(tid=tid, version="mir_eval")] = me_run_time

    # Save the results to a NetCDF file
    result_da.to_netcdf(output_filepath)
    print(f"Timed {tid} and saved to {output_filepath}.")
    return output_filepath


def compare_boundary_metrics(tid, cache_dir="./boundary_metrics", recompute=False):
    salami_hiers = list(fio.salami_ref_hiers(tid=str(tid)).values())
    if len(salami_hiers) <= 1:
        # print(f"Track {tid} has multiple hierarchies, skipping.")
        return

    # Check if already computed
    os.makedirs(cache_dir, exist_ok=True)
    output_filepath = os.path.join(cache_dir, f"b_{tid}.nc")
    if os.path.exists(output_filepath) and not recompute:
        print(f"Already computed {tid}.")
        return xr.open_dataarray(output_filepath)

    adobe_hier = next(iter(fio.salami_adobe_hiers(tid=str(tid)).values()))

    # compute and save results
    result_coords = dict(
        tid=[tid],
        anno_id=[0, 1],
        component=["cap", "ref", "est"],
        event=["beta", "pair", "T"],
        window=[0.5, 3],
    )
    result_da = xr.DataArray(dims=result_coords.keys(), coords=result_coords)
    for anno_id, ref in enumerate(salami_hiers):
        ref = ref.unique_labeling()
        est = adobe_hier.unique_labeling()
        ref_itvls, ref_labels, est_itvls, est_labels = mtr.align_hier(
            ref.itvls, ref.labels, est.itvls, est.labels
        )
        for window in result_coords["window"]:
            # Compute the components
            b_scores = mtr.bmeasure(ref_itvls, est_itvls, trim=False, window=window)
            result_da.loc[
                dict(window=window, anno_id=anno_id, tid=tid, event="beta")
            ] = [b_scores["mu"], b_scores["ref_beta"], b_scores["est_beta"]]
            result_da.loc[
                dict(window=window, anno_id=anno_id, tid=tid, event="pair")
            ] = [b_scores["pairs_cap"], b_scores["ref_pairs"], b_scores["est_pairs"]]
            result_da.loc[dict(window=window, anno_id=anno_id, tid=tid, event="T")] = [
                b_scores["T_mu"],
                b_scores["T_ref_beta"],
                b_scores["T_est_beta"],
            ]

    # Save to NetCDF
    result_da.to_netcdf(output_filepath)
    return result_da
