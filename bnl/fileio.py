import os, jams, json

from . import multi2H, fmt


def get_ref_hiers(tid, salami_jams_dir="/Users/tomxi/data/salami-jams"):
    jams_path = os.path.join(salami_jams_dir, tid + ".jams")
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


def get_adobe_hiers(
    tid,
    result_dir="/Users/tomxi/data/ISMIR21-Segmentations/SALAMI/def_mu_0.1_gamma_0.1/",
):
    filename = f"{tid}.mp3.msdclasscsnmagic.json"

    with open(os.path.join(result_dir, filename), "r") as f:
        adobe_hier = json.load(f)

    anno = fmt.hier2multi(adobe_hier)
    anno.sandbox.update(mu=0.1, gamma=0.1)
    return multi2H(anno)
