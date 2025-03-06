# I want to plot some meet mats for hier annotations returned by mir_eval
# I want to see where the hits and the misses are between two meet matrix.


import bnl, tests
from bnl.utils import gauc
import mir_eval, librosa, os, jams, json
import matplotlib.pyplot as plt
import numpy as np


def make_hierarchies():
    hier1 = bnl.H([tests.ITVLS1, tests.ITVLS2], [tests.LABELS1, tests.LABELS2])
    hier2 = bnl.H(
        [tests.ITVLS3, tests.ITVLS4, tests.ITVLS5],
        [tests.LABELS3, tests.LABELS4, tests.LABELS5],
    )
    hier3 = bnl.H(
        [tests.ITVLS1, tests.ITVLS2, tests.ITVLS3, tests.ITVLS4, tests.ITVLS5],
        [tests.LABELS1, tests.LABELS2, tests.LABELS3, tests.LABELS4, tests.LABELS5],
    )
    return dict(h1=hier1, h2=hier2, h3=hier3)


def make_meet_mat(hier, frame_size=0.1, strict_mono=False):
    """Create meet matrices for a hierarchy with given frame size."""
    hier.update_sr(1 / frame_size)
    return mir_eval.hierarchy._meet(
        hier.itvls, hier.labels, frame_size, strict_mono=strict_mono
    )


def lmeasure_comparison(ref: bnl.H, est: bnl.H, frame_size=0.1):
    """Compare per-frame lmeasure of two hierarchies by computing ranking recall and precision
    for both non-monotonic and monotonic meet matrices.
    """
    results = dict(orig={}, mono={})
    for mode in results.keys():
        strict_mono = True if mode == "mono" else False
        mat_ref = make_meet_mat(ref, frame_size=frame_size, strict_mono=strict_mono)
        mat_est = make_meet_mat(est, frame_size=frame_size, strict_mono=strict_mono)
        recall, rank_recall, norm_recall = gauc(mat_ref, mat_est)
        precision, rank_precision, norm_precision = gauc(mat_est, mat_ref)
        results[mode] = {
            "ref_meet": mat_ref,
            "est_meet": mat_est,
            "l": (precision, recall),
            "q": (rank_precision, rank_recall),
            "norm": (norm_precision, norm_recall),
        }
    return results


def plot_comparison(ref: bnl.H, est: bnl.H, frame_size=0.5):
    """Compare per-frame lmeasure of two hierarchies (with or without strict monotonicity) and plot the results."""
    full_result = lmeasure_comparison(ref, est, frame_size=frame_size)
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(14, 6.5),
        sharex="all",
        sharey="row",
        gridspec_kw={"height_ratios": [4, 1, 1]},
    )

    # Iterate over both modes: original (non-strict) and strict monotonicity
    for strict_mono, offset in zip((False, True), (0, 2)):
        mode = "mono" if strict_mono else "orig"
        # Extract scores and compute time axis
        result = full_result[mode]
        lp, lr = result["l"]
        qp, qr = result["q"]
        norm_p, norm_r = result["norm"]
        ts = np.arange(len(qp)) * frame_size

        # Top row: Plot the reference and estimated meet matrices
        keys = ["ref_meet", "est_meet"]
        titles = [
            f"{mode} reference meet matrix",
            f"{mode} estimated meet matrix",
        ]
        for i, (key, title) in enumerate(zip(keys, titles)):
            ax = axs[0, offset + i]
            librosa.display.specshow(
                result[key].toarray(),
                ax=ax,
                x_axis="time",
                y_axis="time",
                hop_length=1,
                sr=1 / frame_size,
                cmap="gray_r",
            )
            ax.set_title(title)
            ax.set_xlabel("")
            # For non-mono, only clear the ylabel of the second plot
            # For mono, clear ylabels for both plots
            if offset or i:
                ax.set_ylabel("")

        # Second row: Plot norm curves
        axs[1, offset].plot(ts, norm_r)
        axs[1, offset].set_title("total significant pairs in reference")
        axs[1, offset + 1].plot(ts, norm_p)
        axs[1, offset + 1].set_title("total specificity in estimation")

        # Third row: Plot per-frame scores and add horizontal lines
        axs[2, offset].plot(ts, qr)
        axs[2, offset].set_title("per frame recall")
        axs[2, offset].set_xlabel("Time")
        axs[2, offset + 1].plot(ts, qp)
        axs[2, offset + 1].set_title("per frame precision")
        axs[2, offset + 1].set_xlabel("Time")

        axs[2, offset].hlines(
            lr,
            ts[0],
            ts[-1],
            linestyles="dashed",
            color="r",
            label=f"{lr:.2f}",
        )
        axs[2, offset].legend()
        axs[2, offset + 1].hlines(
            lp,
            ts[0],
            ts[-1],
            linestyles="dashed",
            color="r",
            label=f"{lp:.2f}",
        )
        axs[2, offset + 1].legend()

    fig.tight_layout()
    return fig, axs


def get_salami_tids(salami_jams_dir="/Users/tomxi/data/salami-jams"):
    found_jams_files = os.listdir(salami_jams_dir)
    tids = sorted([os.path.splitext(f)[0] for f in found_jams_files])
    return tids


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
        anno_h = bnl.multi2H(bnl.fmt.openseg2multi([upper[anno_id], lower[anno_id]]))
        anno_h_list.append(anno_h)
    return anno_h_list


def get_adobe_hiers(
    tid,
    result_dir="/Users/tomxi/data/ISMIR21-Segmentations/SALAMI/def_mu_0.1_gamma_0.1/",
) -> jams.Annotation:
    filename = f"{tid}.mp3.msdclasscsnmagic.json"

    with open(os.path.join(result_dir, filename), "rb") as f:
        adobe_hier = json.load(f)

    anno = bnl.fmt.hier2multi(adobe_hier)
    anno.sandbox.update(mu=0.1, gamma=0.1)
    return bnl.multi2H(anno)


def save_tmeasure(tid):
    for anno_id, ref_h in enumerate(get_ref_hiers(tid)):
        out_name = f"out/{tid}_{anno_id}_tmeasure.json"
        if os.path.exists(out_name):
            continue
        result = {}

        est_h = get_adobe_hiers(tid)
        ref_h_itvls, est_h_itvls = bnl.utils.pad_itvls(ref_h.itvls, est_h.itvls)
        est_h_mono0 = est_h.force_mono_B(min_seg_dur=0)
        _, est_h_mono0_itvls = bnl.utils.pad_itvls(ref_h.itvls, est_h_mono0.itvls)
        est_h_mono1 = est_h.force_mono_B(min_seg_dur=1)
        _, est_h_mono1_itvls = bnl.utils.pad_itvls(ref_h.itvls, est_h_mono1.itvls)

        # T-measures
        result["orig_r"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls, transitive=False
        )
        result["orig_f"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls, transitive=True
        )
        result["tmeasure1_r"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono1_itvls
        )
        result["tmeasure0_r"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls
        )
        result["tmeasure1_f"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono1_itvls, transitive=True
        )
        result["tmeasure0_f"] = mir_eval.hierarchy.tmeasure(
            ref_h_itvls, est_h_mono0_itvls, transitive=True
        )

        with open(out_name, "w") as f:
            json.dump(result, f)
    return 0


if __name__ == "__main__":
    h_dict = make_hierarchies()
    h3 = h_dict["h3"]
    h2 = h_dict["h2"]
    h1 = h_dict["h1"]
    # hier_ref, hier_est = list(make_hierarchies().values())
    # fig, axs = plot_comparison(hier_ref, hier_est, frame_size=0.1)
    # fig.savefig("scripts/figs/meet_mats_compare_both.pdf")
    # fig, axs = h3.plot(text=True, relabel=False)
    # fig.savefig("scripts/figs/h3.pdf")
    h3_mono_b = h3.force_mono_B()
    # fig, axs = h3_mono_b.plot(text=True, relabel=False)
    # fig.savefig("scripts/figs/h3_mono_b.pdf")
    # h3_mono_l = h3.force_mono_L()
    # fig, axs = h3_mono_l.plot(text=True, relabel=False, legend_ncol=5)
    # fig.savefig("scripts/figs/h3_mono_l.pdf")
    # fig, axs = h3_mono_l.plot(text=True, relabel=True)
    # fig.savefig("scripts/figs/h3_mono_l_relabel.pdf")

    # Now I want to look at how T-measure's are doing.
    print(
        mir_eval.hierarchy.tmeasure(
            h1.itvls,
            h2.itvls,
            transitive=False,
        )
    )

    print(
        mir_eval.hierarchy.tmeasure(
            h1.itvls,
            h2.itvls,
            transitive=True,
        )
    )

    print(
        mir_eval.hierarchy.tmeasure(
            h1.itvls,
            h3.itvls,
            transitive=False,
        )
    )

    print(
        mir_eval.hierarchy.tmeasure(
            h1.itvls,
            h3.itvls,
            transitive=True,
        )
    )

    print(
        mir_eval.hierarchy.tmeasure(
            h1.itvls,
            h3_mono_b.itvls,
            transitive=False,
        )
    )
    print(
        mir_eval.hierarchy.tmeasure(
            h1.itvls,
            h3_mono_b.itvls,
            transitive=True,
        )
    )
