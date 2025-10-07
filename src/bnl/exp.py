import itertools
import os

import mir_eval
import pandas as pd

from . import metrics


def tmeasure_mono_casting_effects(
    slm_track, save_folder="./monocasting_results_tmeasure/", overwrite=False
):
    # Construct the expected output file path
    file_name = f"{slm_track.track_id}.feather"
    output_path = os.path.join(save_folder, file_name)

    # If not overwriting and the file exists, load and return it
    if not overwrite and os.path.exists(output_path):
        print(f"Loading existing results for track {slm_track.track_id} from {output_path}")
        return pd.read_feather(output_path)

    try:
        # Attempt to get the first reference
        ref = list(slm_track.refs.values())[0]
        # Attempt to get a specific estimate and align it with the ref
        raw_est = slm_track.ests["mu1gamma9"].align(ref)
    except (IndexError, KeyError):
        # If either ref or est is not found, print a message and exit
        print(f"--> Error: Can't find ref or est for track {slm_track.track_id}. Skipping.")
        return None

    print(f"Running experiment for track {slm_track.track_id}...")
    params = {
        "prom_func": ["depth", "prob"],
        "bdry_cleaning": ["absorb", "kde", "none"],
        "leveling": ["unique", "mean_shift"],
    }
    param_combos = list(itertools.product(*params.values()))
    results_list = []

    for prom_func, bdry_cleaning, leveling in param_combos:
        processed_est = (
            raw_est.contour(prom_func)
            .clean(bdry_cleaning)
            .level(leveling)
            .to_ms(name=" ".join([prom_func, bdry_cleaning, leveling]))
            .scrub_labels()
        )

        t_reduced_p, t_reduced_r, t_reduced_f = mir_eval.hierarchy.tmeasure(
            ref.itvls, processed_est.itvls
        )
        t_full_p, t_full_r, t_full_f = mir_eval.hierarchy.tmeasure(
            ref.itvls, processed_est.itvls, transitive=True
        )

        record = {
            "track_id": slm_track.track_id,
            "prom_func": prom_func,
            "bdry_cleaning": bdry_cleaning,
            "leveling": leveling,
            "num_est_lvl": len(processed_est),
            "num_est_bs": len(processed_est[-1]) - 1,
            "t_reduced_p": t_reduced_p,
            "t_reduced_r": t_reduced_r,
            "t_reduced_f": t_reduced_f,
            "t_full_p": t_full_p,
            "t_full_r": t_full_r,
            "t_full_f": t_full_f,
        }
        results_list.append(record)

    results_df = pd.DataFrame(results_list)
    os.makedirs(save_folder, exist_ok=True)
    results_df.to_feather(output_path)

    print(f"Results for track {slm_track.track_id} saved to {output_path}")

    return results_df


def bmeasure_mono_casting_effects(
    slm_track, save_folder="./monocasting_results_bmeasure/", overwrite=False, verbose=False
):
    # Construct the expected output file path
    file_name = f"{slm_track.track_id}.feather"
    output_path = os.path.join(save_folder, file_name)

    # If not overwriting and the file exists, load and return it
    if not overwrite and os.path.exists(output_path):
        if verbose:
            print(f"Loading existing results for track {slm_track.track_id} from {output_path}")
        return pd.read_feather(output_path)

    try:
        # Attempt to get the first reference
        ref = list(slm_track.refs.values())[0]
        # Attempt to get a specific estimate and align it with the ref
        raw_est = slm_track.ests["mu1gamma9"].align(ref)
    except (IndexError, KeyError):
        # If either ref or est is not found, print a message and exit
        print(f"--> Error: Can't find ref or est for track {slm_track.track_id}. Skipping.")
        return None

    if verbose:
        print(f"Running experiment for track {slm_track.track_id}...")
    params = {
        "prom_func": ["depth", "prob"],
        "bdry_cleaning": ["absorb", "kde", "none"],
        "leveling": ["unique", "mean_shift"],
    }
    param_combos = itertools.product(*params.values())
    records = []

    for prom_func, bdry_cleaning, leveling in param_combos:
        est_bc = raw_est.contour(prom_func).clean(bdry_cleaning).level(leveling)
        ref_bc = ref.contour("depth").level()

        bmeasure_df = (
            metrics.bmeasure_suite(ref_bc, est_bc, track_id=slm_track.track_id)
            .pivot_table(index=["track_id", "prf", "window"], columns=["metric"], values="score")
            .reset_index()
        )
        bmeasure_df["prom_func"] = prom_func
        bmeasure_df["bdry_cleaning"] = bdry_cleaning
        bmeasure_df["leveling"] = leveling
        bmeasure_df["window"] = bmeasure_df.window.astype("str")
        bmeasure_df["num_est_bs"] = len(est_bc)
        col_order = [
            "track_id",
            "prom_func",
            "bdry_cleaning",
            "leveling",
            "num_est_bs",
            "prf",
            "window",
            "b",
            "b-m",
            "hr",
            "poa",
            "poa-m",
        ]
        records.append(bmeasure_df[col_order])

    results_df = pd.concat(records).reset_index(drop=True)
    results_df.columns.name = None
    os.makedirs(save_folder, exist_ok=True)
    results_df.to_feather(output_path)
    if verbose:
        print(f"Results for track {slm_track.track_id} saved to {output_path}")
    return results_df


def bmeasure_between_slm_refs(track):
    """Runs bmeasure between the first two references in a track."""
    if len(track.refs) < 2:
        raise ValueError(f"Track {track.track_id} has less than 2 references.")
    ref1, ref2 = track.refs.values()
    return metrics.bmeasure_suite(
        ref1.contour("depth"),
        ref2.contour("depth"),
        track_id=track.track_id,
    )


def mir_eval_between_slm_refs(track):
    """Runs mir_eval.hierarchy.evaluate between the first two references in a track."""
    if len(track.refs) < 2:
        raise ValueError(f"Track {track.track_id} has less than 2 references.")
    ref1, ref2 = track.refs.values()
    record = mir_eval.hierarchy.evaluate(ref1.itvls, ref1.labels, ref2.itvls, ref2.labels)
    record["track_id"] = track.track_id
    return pd.Series(record)


def mir_eval_flat_between_slm_refs(track):
    if len(track.refs) < 2:
        raise ValueError(f"Track {track.track_id} has less than 2 references.")
    ref1, ref2 = track.refs.values()
    upper_hr = mir_eval.segment.detection(ref1.itvls[0], ref2.itvls[0])
    lower_hr = mir_eval.segment.detection(ref1.itvls[1], ref2.itvls[1])
    record = {
        "track_id": track.track_id,
        "upper_hr": upper_hr[2],
        "lower_hr": lower_hr[2],
    }
    return pd.Series(record)
