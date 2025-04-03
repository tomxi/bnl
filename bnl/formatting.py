import jams
import numpy as np
from collections import defaultdict


def multi2hier(anno) -> list:
    n_lvl_list = [obs.value["level"] for obs in anno]
    n_lvl = max(n_lvl_list) + 1
    hier = [[[], []] for _ in range(n_lvl)]
    for obs in anno:
        lvl = obs.value["level"]
        label = obs.value["label"]
        interval = [obs.time, obs.time + obs.duration]
        hier[lvl][0].append(interval)
        hier[lvl][1].append(label)

    for i in range(n_lvl):
        hier[i][0] = np.array(hier[i][0])
    return hier


def hier2multi(hier) -> jams.Annotation:
    anno = jams.Annotation(namespace="multi_segment")
    anno.duration = hier[0][0][-1][-1]
    anno.time = hier[0][0][0][0]
    for layer, (intervals, labels) in enumerate(hier):
        for ival, label in zip(intervals, labels):
            anno.append(
                time=ival[0],
                duration=ival[1] - ival[0],
                value={"label": label, "level": layer},
            )
    return anno


def hier2mireval(hier) -> tuple:
    intervals = []
    labels = []
    for itv, lbl in hier:
        intervals.append(itv)
        labels.append(lbl)

    return [col for col in zip(*hier)]


def mireval2hier(itvls: list, labels: list) -> list:
    hier = []
    n_lvl = len(labels)
    for lvl in range(n_lvl):
        lvl_anno = [itvls[lvl], labels[lvl]]
        hier.append(lvl_anno)
    return hier


def multi2mireval(anno) -> tuple:
    return hier2mireval(multi2hier(anno))


def mireval2multi(itvls: list, labels: list) -> jams.Annotation:
    return hier2multi(mireval2hier(itvls, labels))


def openseg2multi(annos: list) -> jams.Annotation:
    multi_anno = jams.Annotation(namespace="multi_segment")
    longest_duration = 0
    for lvl, openseg in enumerate(annos):
        if openseg.duration > longest_duration:
            longest_duration = openseg.duration
        for obs in openseg:
            multi_anno.append(
                time=obs.time,
                duration=obs.duration,
                value={"label": obs.value, "level": lvl},
            )
    multi_anno.duration = longest_duration
    return multi_anno


def multi2mirevalflat(multi_anno, layer=-1):
    all_itvls, all_labels = multi2mireval(multi_anno)
    return all_itvls[layer], all_labels[layer]


def mirevalflat2openseg(itvls, labels):
    anno = jams.Annotation(namespace="segment_open")
    duration = itvls[-1][-1]
    for ival, label in zip(itvls, labels):
        anno.append(
            time=ival[0],
            duration=ival[1] - ival[0],
            value=label,
        )
    anno.duration = duration
    return anno


def multi2openseg(multi_anno, layer=-1):
    itvls, labels = multi2mirevalflat(multi_anno, layer)
    return mirevalflat2openseg(itvls, labels)


def openseg2mirevalflat(openseg_anno):
    return multi2mirevalflat(openseg2multi([openseg_anno]))


# def bs_dict2itvls(bs_dict):
#     # group the bs_dict keys by distinct values
#     # and sort them by the values
#     bs_by_sal = defaultdict(list)
#     max_sal = max(bs_dict.values())
#     for b, sal in bs_dict.items():
#         bs_by_sal[sal].append(b)

#     # sort by keys:
#     sals, bs_lists = sorted(bs_by_sal.items())

#     bs_per_lvl = []
#     for sal in range(max_sal):
#         bs_per_lvl.append()
#     for key, value in bs_by_sal.items():
#         itvls.append([key, key + value])
#     return itvls
