import os, jams, json
from . import multi2H, fmt, H

ROOT_DATA_DIR = os.path.expanduser("~/data/")


def salami_ref_hiers(tid, salami_jams_dir="salami-jams/"):
    jams_path = os.path.join(ROOT_DATA_DIR, salami_jams_dir, str(tid) + ".jams")
    jam = jams.load(jams_path)
    duration = jam.file_metadata.duration
    upper = jam.search(namespace="segment_salami_upper")
    upper_annotators = [anno.annotation_metadata.annotator.name for anno in upper]
    lower = jam.search(namespace="segment_salami_lower")
    lower_annotators = [anno.annotation_metadata.annotator.name for anno in lower]
    ref_hiers = dict()
    for anno_id in range(len(upper)):
        upper[anno_id].duration = duration
        lower[anno_id].duration = duration
        upper_annotator = upper_annotators[anno_id]
        lower_annotator = lower_annotators[anno_id]
        if upper_annotator != lower_annotator:
            raise ValueError(
                f"Upper and lower annotators do not match: {upper_annotator} vs {lower_annotator}"
            )
        # Convert to multi2H format
        ref_hiers[upper_annotator] = multi2H(
            fmt.openseg2multi([upper[anno_id], lower[anno_id]])
        )
    return ref_hiers


def salami_adobe_hiers(
    tid,
    result_dir="ISMIR21-Segmentations/SALAMI/",
):
    filename = f"{tid}.mp3.msdclasscsnmagic.json"

    options = ["def_mu_0.1_gamma_0.1", "def_mu_0.5_gamma_0.5", "def_mu_0.1_gamma_0.9"]
    opt_strs = [s.replace("def_", "").replace("_0.", "") for s in options]
    hiers = dict()
    for opt, opt_str in zip(options, opt_strs):
        with open(os.path.join(ROOT_DATA_DIR, result_dir, opt, filename), "r") as f:
            raw_hierarchy = json.load(f)
            # Raw_hierarchy is list of [itvls, labels], H need list of itvls and list of labels.
            # zip and unpack
            hiers[opt_str] = H(*zip(*raw_hierarchy))
    return hiers


def salami_tids(salami_jams_dir="salami-jams"):
    found_jams_files = os.listdir(os.path.join(ROOT_DATA_DIR, salami_jams_dir))
    tids = sorted([os.path.splitext(f)[0] for f in found_jams_files])
    return tids


def salami_annos(slm_tid):
    return salami_ref_hiers(slm_tid), salami_adobe_hiers(slm_tid)
