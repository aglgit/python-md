import sys
import numpy as np
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import make_symmetry_functions

sys.path.insert(1, "../tools")

from analysis import Analyzer
from plotting import Plotter
from training import Trainer


if __name__ == "__main__":
    train_traj = "trajs/training.traj"
    cutoff = Polynomial(6.0, gamma=5.0)
    elements = ["Cu"]
    num_radial_etas = 7
    num_angular_etas = 10
    num_zetas = 1
    angular_type = "G4"
    symm_funcs = {}

    trn = Trainer(cutoff=cutoff)
    trn.create_Gs(elements, num_radial_etas, num_angular_etas, num_zetas, angular_type)
    symm_funcs["Selected"] = trn.Gs

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

    anl = Analyzer()
    plter = Plotter()
    r, rdf = anl.calculate_rdf(train_traj, r_max=cutoff.Rc)

    for label, symm_func in symm_funcs.items():
        plter.plot_symmetry_functions(
            label + "_rad.png",
            label + "_ang.png",
            symm_func,
            rij=r,
            rdf=rdf,
            cutoff=cutoff,
        )
