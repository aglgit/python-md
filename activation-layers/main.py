import sys
import numpy as np
from asap3 import EMT
from amp.descriptor.cutoffs import Polynomial
from amp.descriptor.gaussian import make_symmetry_functions

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 500

    n_train = int(8e2)
    n_test = int(2e2)
    save_interval = 100

    max_steps = int(5e2)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    cutoff = Polynomial(4.0)

    elements = ["Cu"]
    gammas = [1.0, -1.0]
    nr = 4
    nz = 1
    radial_etas = 10.0 * np.ones(nr)
    centers = np.linspace(0.5, cutoff.Rc + 0.5, nr)
    angular_etas = np.linspace(0.1, 1.0, nr)
    zetas = [4 ** i for i in range(nz)]

    G2 = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )
    G5 = make_symmetry_functions(
        elements=elements, type="G5", etas=angular_etas, zetas=zetas, gammas=[1.0, -1.0]
    )
    Gs = G2 + G5

    trjbd = TrajectoryBuilder()
    calc = EMT()
    train_atoms = trjbd.build_atoms(system, size, temp, calc)
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc)

    train_traj = "training.traj"
    test_traj = "test.traj"
    steps, train_traj = trjbd.integrate_atoms(
        train_atoms, train_traj, n_train, save_interval
    )
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval
    )

    activation = ["tanh", "sigmoid"]
    hidden_layers = [[10], [20], [30], [40], [10, 10], [20, 10], [30, 10], [40, 40]]
    calcs = {}
    for ac in activation:
        for hl in hidden_layers:
            trn = Trainer(
                convergence=convergence,
                force_coefficient=force_coefficient,
                activation=ac,
                hidden_layers=hl,
                cutoff=cutoff,
            )
            label = "{}-{}".format(ac, hl)
            calc = trn.create_calc(label=label, dblabel=label)
            amp_name = trn.train_calc(calc, train_traj)
            calcs[label] = amp_name

    columns = ["Activation/Hidden layers", "Energy RMSE", "Force RMSE"]
    trn.test_calculators(calcs, test_traj, columns)
