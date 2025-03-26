import os, jams, json, mir_eval
from . import multi2H, fmt, utils

ROOT_DATA_DIR = "/Users/tomxi/data/"


def salami_ref_hiers(tid, salami_jams_dir="salami-jams/"):
    jams_path = os.path.join(ROOT_DATA_DIR, salami_jams_dir, tid + ".jams")
    jam = jams.load(jams_path)
    duration = jam.file_metadata.duration
    upper = jam.search(namespace="segment_salami_upper")
    lower = jam.search(namespace="segment_salami_lower")
    anno_h_list = []
    for anno_id in range(len(upper)):
        upper[anno_id].duration = duration
        lower[anno_id].duration = duration
        anno_h = multi2H(fmt.openseg2multi([upper[anno_id], lower[anno_id]]))
        anno_h_list.append(anno_h)
    return anno_h_list


def adobe_hiers(
    tid,
    result_dir="ISMIR21-Segmentations/SALAMI/def_mu_0.1_gamma_0.1/",
):
    filename = f"{tid}.mp3.msdclasscsnmagic.json"

    with open(os.path.join(ROOT_DATA_DIR, result_dir, filename), "r") as f:
        adobe_hier = json.load(f)

    anno = fmt.hier2multi(adobe_hier)
    anno.sandbox.update(mu=0.1, gamma=0.1)
    return multi2H(anno)


def salami_tids(salami_jams_dir="salami-jams"):
    found_jams_files = os.listdir(os.path.join(ROOT_DATA_DIR, salami_jams_dir))
    tids = sorted([os.path.splitext(f)[0] for f in found_jams_files])
    return tids


def save_tmeasure(tid):
    for anno_id, ref_h in enumerate(salami_ref_hiers(tid)):
        out_name = f"out/{tid}_{anno_id}_tmeasure.json"
        if os.path.exists(out_name):
            continue
        result = {}

        est_h = adobe_hiers(tid)
        ref_h_itvls, est_h_itvls = utils.pad_itvls(ref_h.itvls, est_h.itvls)
        est_h_mono0 = est_h.force_mono_B(min_seg_dur=0)
        _, est_h_mono0_itvls = utils.pad_itvls(ref_h.itvls, est_h_mono0.itvls)
        est_h_mono1 = est_h.force_mono_B(min_seg_dur=1)
        _, est_h_mono1_itvls = utils.pad_itvls(ref_h.itvls, est_h_mono1.itvls)

        # T-measures
        result["orig_r"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls, transitive=False
        )
        result["orig_f"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls, transitive=True
        )
        result["mono1_r"] = mir_eval.hierarchy.tmeasure(ref_h_itvls, est_h_mono1_itvls)
        result["mono0_r"] = mir_eval.hierarchy.tmeasure(ref_h_itvls, est_h_mono0_itvls)
        result["mono1_f"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono1_itvls, transitive=True
        )
        result["mono0_f"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls, transitive=True
        )

        with open(out_name, "w") as f:
            json.dump(result, f)
    return 0
