import jams
import numpy as np
from typing import List, Tuple, Any


def multi2hier(
    anno: jams.Annotation,
) -> List[Tuple[List[List[List[float]]], List[List[str]]]]:
    n_lvl_list = [obs.value["level"] for obs in anno]
    n_lvl = max(n_lvl_list) + 1
    hier = [[[], []] for _ in range(n_lvl)]
    for obs in anno:
        lvl = obs.value["level"]
        label = obs.value["label"]
        interval = [round(obs.time, 3), round(obs.time + obs.duration, 3)]
        hier[lvl][0].append(interval)
        hier[lvl][1].append(str(label))
    return hier


def hier2multi(
    hier: List[Tuple[List[List[List[float]]], List[List[str]]]]
) -> jams.Annotation:
    anno = jams.Annotation(namespace="multi_segment")
    # Assuming hier[0][0] is non-empty; otherwise, adjust accordingly.
    anno.time = hier[0][0][0][0]
    anno.duration = hier[0][0][-1][-1]
    for layer, (intervals, labels) in enumerate(hier):
        for ival, label in zip(intervals, labels):
            anno.append(
                time=round(ival[0], 3),
                duration=round(ival[1] - ival[0], 3),
                value={"label": str(label), "level": layer},
            )
    return anno


def hier2mireval(
    hier: List[Tuple[List[List[List[float]]], List[List[str]]]]
) -> Tuple[List[np.ndarray], List[str]]:
    intervals: List[np.ndarray] = []
    labels: List[str] = []
    for itv, lbl in hier:
        intervals.append(np.array(itv, dtype=float))
        labels.append(lbl)
    return intervals, labels


def mireval2hier(itvls: List[List[float]], labels: List[str]) -> List:
    hier = []
    n_lvl = len(labels)
    for lvl in range(n_lvl):
        lvl_anno = [itvls[lvl], labels[lvl]]
        hier.append(lvl_anno)
    return hier


def multi2mireval(
    anno: jams.Annotation,
) -> Tuple[List[List[List[float]]], List[List[str]]]:
    return hier2mireval(multi2hier(anno))


def mireval2multi(
    itvls: List[List[List[float]]], labels: List[List[str]]
) -> jams.Annotation:
    return hier2multi(mireval2hier(itvls, labels))


def openseg2multi(annos: List[jams.Annotation]) -> jams.Annotation:
    multi_anno = jams.Annotation(namespace="multi_segment")

    for lvl, openseg in enumerate(annos):
        for obs in openseg:
            multi_anno.append(
                time=round(obs.time, 3),
                duration=round(obs.duration, 3),
                value={"label": obs.value, "level": lvl},
            )
    return multi_anno


def multi2mirevalflat(
    multi_anno: jams.Annotation, layer: int = -1
) -> Tuple[List[List[List[float]]], List[List[str]]]:
    all_itvls, all_labels = multi2mireval(multi_anno)
    return all_itvls[layer], all_labels[layer]


def multi2openseg(multi_anno: jams.Annotation, layer: int = -1) -> jams.Annotation:
    itvls, labels = multi2mirevalflat(multi_anno, layer)
    anno = jams.Annotation(namespace="segment_open")
    for ival, label in zip(itvls, labels):
        anno.append(
            time=round(ival[0], 3),
            duration=round(ival[1] - ival[0], 3),
            value=str(label),
        )
    return anno


def openseg2mirevalflat(
    openseg_anno: jams.Annotation,
) -> Tuple[List[List[List[float]]], List[List[str]]]:
    return multi2mirevalflat(openseg2multi([openseg_anno]))
