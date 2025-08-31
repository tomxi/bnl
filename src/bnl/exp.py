import itertools
import os
import random

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
        "mono_casting": ["depth", "prob"],
        "bdry_cleaning": ["absorb", "kde", "none"],
        "leveling": ["unique", "mean_shift"],
    }
    param_combos = list(itertools.product(*params.values()))
    results_list = []

    for mono_casting, bdry_cleaning, leveling in param_combos:
        processed_est = (
            raw_est.contour(mono_casting)
            .clean(bdry_cleaning)
            .level(leveling)
            .to_ms(name=" ".join([mono_casting, bdry_cleaning, leveling]))
            .scrub_labels()
        )

        t_reduced_p, t_reduced_r, t_reduced_f = mir_eval.hierarchy.tmeasure(
            ref.itvls, processed_est.itvls
        )
        t_full_p, t_full_r, t_full_f = mir_eval.hierarchy.tmeasure(
            ref.itvls, processed_est.itvls, transitive=True
        )

        record = {
            "tid": slm_track.track_id,
            "mono_casting": mono_casting,
            "bdry_cleaning": bdry_cleaning,
            "leveling": leveling,
            "num_levels": len(processed_est),
            "num_boundaries": len(processed_est[-1]) - 1,
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
    slm_track, save_folder="./monocasting_results_bmeasure/", overwrite=False
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
        "mono_casting": ["depth", "prob"],
        "bdry_cleaning": ["absorb", "kde", "none"],
        "leveling": ["unique", "mean_shift"],
    }
    param_combos = list(itertools.product(*params.values()))
    results_list = []

    for mono_casting, bdry_cleaning, leveling in param_combos:
        est_bc = raw_est.contour(mono_casting).clean(bdry_cleaning).level(leveling)
        ref_bc = ref.contour("depth").level()

        for window, reduced in [
            (0.5, False),
            (0.5, True),
            (3, False),
            (3, True),
        ]:
            raw_data = metrics.bmeasure(
                ref_bc, est_bc, match_window=window, reduced=reduced, return_raw_data=True
            )
            record = {
                "tid": slm_track.track_id,
                "mono_casting": mono_casting,
                "bdry_cleaning": bdry_cleaning,
                "leveling": leveling,
                "window": window,
                "reduced": reduced,
                "bhr_num": raw_data["hr_num"],
                "bhr_rec_denom": raw_data["hr_recall_denom"],
                "bhr_prec_denom": raw_data["hr_precision_denom"],
                "bpo_rec_num": raw_data["po_recall_num"],
                "bpo_rec_denom": raw_data["po_recall_denom"],
                "bpo_prec_num": raw_data["po_precision_num"],
                "bpo_prec_denom": raw_data["po_precision_denom"],
            }
            results_list.append(record)

    results_df = pd.DataFrame(results_list)
    os.makedirs(save_folder, exist_ok=True)
    results_df.to_feather(output_path)

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
        trim=True,
    )


def mir_eval_between_slm_refs(track):
    """Runs mir_eval.hierarchy.evaluate between the first two references in a track."""
    if len(track.refs) < 2:
        raise ValueError(f"Track {track.track_id} has less than 2 references.")
    ref1, ref2 = track.refs.values()
    record = mir_eval.hierarchy.evaluate(ref1.itvls, ref1.labels, ref2.itvls, ref2.labels)
    record["track_id"] = track.track_id
    return pd.Series(record)
