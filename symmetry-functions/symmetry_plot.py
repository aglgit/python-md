import sys
import numpy as np
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import make_symmetry_functions

sys.path.insert(1, "../tools")

from analysis import Analyzer
from plotting import Plotter


if __name__ == "__main__":
    train_traj = "trajs/training.traj"

    symm_funcs = {}
    elements = ["Cu"]
    cutoff = Polynomial(6.0, gamma=5.0)

    G2 = make_symmetry_functions(
        elements=elements, type="G2", etas=[0.05, 0.23, 1.0, 5.0], centers=np.zeros(4)
    )
    G4 = make_symmetry_functions(
        elements=elements,
        type="G4",
        etas=0.005 * np.ones(1),
        zetas=[1.0, 4.0],
        gammas=[1.0, -1.0],
    )
    symm_funcs["Default"] = G2 + G4

    nr = 6
    nz = 1
    radial_etas = np.linspace(1.0, 20.0, nr)
    centers = np.zeros(nr)
    G2_uncentered = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )

    radial_etas = 5.0 * np.ones(nr)
    centers = np.linspace(0.5, cutoff.Rc - 0.5, nr)
    G2_centered = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )

    angular_etas = np.linspace(0.01, 3.0, nr + 10)
    zetas = [2 ** i for i in range(nz)]
    G4 = make_symmetry_functions(
        elements=elements, type="G4", etas=angular_etas, zetas=zetas, gammas=[1.0, -1.0]
    )
    symm_funcs["Selected"] = G2_uncentered + G2_centered + G4

    anl = Analyzer()
    plter = Plotter()
    r, rdf = anl.calculate_rdf(train_traj, r_max=cutoff.Rc)

    for label, symm_func in symm_funcs.items():
        plter.plot_symmetry_functions(
            label + "-rad.png",
            label + "-ang.png",
            symm_func,
            rij=r,
            rdf=rdf,
            cutoff=cutoff,
        )
