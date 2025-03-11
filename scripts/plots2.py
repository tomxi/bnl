import bnl, tests
from bnl import fio, viz
import matplotlib.pyplot as plt
import numpy as np


def create_fig(figsize=(3.5, 5)):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    return fig, [ax1, ax2, ax3]


if __name__ == "__main__":
    fig, axs = create_fig()
    axs[0].plot([0, 1], [0, 1])
    axs[1].scatter([0, 1], [1, 0])
    axs[2].set_title("Combined Plot")
    axs[2].legend(["Line", "Scatter"])
    fig.savefig("scripts/figs/fig1.pdf")
