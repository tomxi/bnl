import bnl, tests
from bnl import fio, viz


if __name__ == "__main__":
    hiers = tests.make_hierarchies()
    fig, axs = viz.create_grid_fig()
    hier = hiers["h2"]
    for i, seg in enumerate(hier.levels[:-1]):
        seg.plot(ax=axs[i][0], text=True, ytick=i + 1, time_ticks=False)

    hier.levels[-1].plot(
        ax=axs[hier.d - 1][0], text=True, ytick=hier.d, time_ticks=True
    )

    fig.savefig("scripts/figs/fig1.pdf", transparent=True)
