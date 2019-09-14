import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, "../tools")

from analysis import Analyzer
from plotting import Plotter
from training import Trainer


if __name__ == "__main__":
    sns.set()
    plot_dir = "plots"
    plot_file = os.path.join(plot_dir, "rdf.png")
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    anl = Analyzer()
    plter = Plotter()
    r_cut = 6.0
    r, rdf = anl.calculate_rdf("trajs/training.traj", r_max=r_cut)
    rdf[np.nonzero(rdf)] /= max(rdf)
    cutoff = plter.polynomial(r, r_cut, gamma=5.0)

    plt.plot(r, rdf, label="Radial distribution function")
    plt.plot(r, cutoff, label="Polynomial cutoff, gamma=5.0")
    plt.legend()
    plt.title("Copper radial distribution function")
    plt.xlabel("Radial distance [Angstrom]")
    plt.ylabel("Radial distribution function (normalized to 1)")
    plt.savefig(plot_file) 
