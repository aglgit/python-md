import sys

sys.path.insert(0, "../tools")

import numpy as np
from analysis import Analyzer
from plot import Plotter


if __name__ == "__main__":

    test = {
        "Gs-4-4-G4": [
            {"type": "G2", "element": "Cu", "eta": 0.1},
            {"type": "G2", "element": "Cu", "eta": 1.0},
            {"type": "G2", "element": "Cu", "eta": 2.5},
            {"type": "G2", "element": "Cu", "eta": 5.0},
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": 1.0,
                "zeta": 1,
            },
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": -1.0,
                "zeta": 1,
            },
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": 1.0,
                "zeta": 2,
            },
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": -1.0,
                "zeta": 2,
            },
        ]
    }

    anl = Analyzer(save_interval=100)
    plt = Plotter()
    r_cut = 6.0
    rij, rdf = anl.calculate_rdf("trajs/training.traj", rmax=r_cut)
    plt.plot_symmetry_functions("", "", test, r_cut=r_cut, rij=rij, rdf=rdf)
