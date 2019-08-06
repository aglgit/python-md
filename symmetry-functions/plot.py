import sys
import numpy as np
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import make_symmetry_functions

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    cutoff = Polynomial(5.0)
    elements = ["Cu"]
    nr = 4
    nz = 1
    gammas = [1.0, -1.0]
    for shift in ["uncentered", "centered"]:
        if shift == "uncentered":
            radial_etas = np.logspace(np.log10(0.5), np.log10(80.0), nr)
            centers = np.zeros(nr)
        elif shift == "centered":
            radial_etas = 10.0 * np.ones(nr)
            centers = np.linspace(0.5, cutoff.Rc + 0.5, nr)
        angular_etas = np.linspace(0.1, 1.0, nr)
        zetas = [4 ** i for i in range(nz)]

        G2 = make_symmetry_functions(
            elements=elements, type="G2", etas=radial_etas, centers=centers
        )

        G5 = make_symmetry_functions(
            elements=elements,
            type="G5",
            etas=angular_etas,
            zetas=zetas,
            gammas=[1.0, -1.0],
        )
        Gs = G2 + G5

        plter = Plotter()
        plter.plot_symmetry_functions(
            "radial_{}.png".format(shift),
            "angular_{}.png".format(shift),
            Gs,
            cutoff=cutoff,
        )
