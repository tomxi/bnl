import bnl, tests
from bnl import fio, viz


if __name__ == "__main__":
    hiers = tests.make_hierarchies()
    fig, axs = hiers["h2"].plot()

    fig.savefig("scripts/figs/fig1.pdf", transparent=True)
