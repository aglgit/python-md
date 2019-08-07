import sys
import numpy as np
from asap3 import EMT
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine
from amp.descriptor.gaussian import make_symmetry_functions

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 500

    n_test = int(2e4)
    save_interval = 100

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Cosine(6.0)

    elements = ["Cu"]
    nr = 4
    nz = 1
    radial_etas = np.logspace(np.log10(1.0), np.log10(20.0), nr)
    centers = np.zeros(nr)
    G2_uncentered = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )

    radial_etas = np.logspace(np.log10(5.0), np.log10(20.0), nr)
    centers = np.linspace(1.0, cutoff.Rc - 1.0, nr)
    G2_centered = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )
    G2 = G2_uncentered + G2_centered

    angular_etas = np.linspace(0.05, 1.0, 8)
    zetas = [4 ** i for i in range(nz)]
    G4 = make_symmetry_functions(
        elements=elements, type="G4", etas=angular_etas, zetas=zetas, gammas=[1.0, -1.0]
    )
    Gs = G2 + G4

    trn = Trainer(
        convergence=convergence,
        force_coefficient=force_coefficient,
        hidden_layers=hidden_layers,
        activation=activation,
        cutoff=cutoff,
        Gs=Gs,
    )

    trjbd = TrajectoryBuilder()
    n_images = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    train_trajs = ["training_n{}.traj".format(ni) for ni in n_images]
    for i in range(len(n_images)):
        calc = EMT()
        train_atoms = trjbd.build_atoms(system, size, temp, calc)
        n_train = n_images[i] * save_interval
        steps, train_trajs[i] = trjbd.integrate_atoms(
            train_atoms, train_trajs[i], n_train, save_interval
        )

    test_traj = "test.traj"
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc)
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval
    )

    calcs = {}
    for i in range(len(n_images)):
        label = "n{}".format(n_images[i])
        calc = trn.create_calc(label=label, dblabel=label)
        ann = Annealer(
            calc=calc,
            images=train_trajs[i],
            Tmax=20,
            Tmin=1,
            steps=4000,
            train_forces=False,
        )
        amp_name = trn.train_calc(calc, train_trajs[i])
        calcs[label] = amp_name

    columns = ["Number of images", "Energy RMSE", "Force RMSE"]
    trn.test_calculators(calcs, test_traj, columns)
