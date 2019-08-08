import os
import sys
import numpy as np
from asap3 import EMT
from amp.utilities import Annealer
from amp.analysis import calculate_error
from amp.descriptor.cutoffs import Cosine
from amp.descriptor.gaussian import make_symmetry_functions

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    system = "copper"
    size = (3, 3, 3)
    temp = 500

    n_train = int(5e4)
    n_test = int(1e4)
    save_interval = 100

    max_steps = int(2e3)
    activation = "tanh"
    hidden_layers = [10, 10]
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

    plter = Plotter()
    energy_noforcetrain = "energy_noforcetrain.png"
    force_noforcetrain = "force_noforcetrain.png"
    energy_forcetrain = "energy_forcetrain.png"
    force_forcetrain = "force_forcetrain.png"

    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    trn_energy = Trainer(
        convergence=convergence,
        force_coefficient=force_coefficient,
        activation=activation,
        hidden_layers=hidden_layers,
        cutoff=cutoff,
        Gs=Gs,
    )
    label = "energy"
    dblabel = label + "-train"
    calc = trn_energy.create_calc(label=label, dblabel=dblabel)
    ann = Annealer(
        calc=calc, images=train_traj, Tmax=20, Tmin=1, steps=2000, train_forces=False
    )
    energy_amp_name = trn_energy.train_calc(calc, train_traj)
    dblabel = label + "-test"
    energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
        energy_amp_name, images=test_traj, label=label, dblabel=dblabel
    )
    plter.plot_amp_error(
        energy_noforcetrain,
        force_noforcetrain,
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )

    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    force_coefficient = 0.1
    trn_force = Trainer(
        convergence=convergence,
        force_coefficient=force_coefficient,
        activation=activation,
        hidden_layers=hidden_layers,
        cutoff=cutoff,
        Gs=Gs,
    )
    label = "force"
    dblabel = label + "-train"
    calc = trn_force.create_calc(label=label, dblabel=dblabel)
    ann = Annealer(
        calc=calc, images=train_traj, Tmax=20, Tmin=1, steps=4000, train_forces=False
    )
    force_amp_name = trn_force.train_calc(calc, train_traj)
    dblabel = label + "-test"
    energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
        force_amp_name, images=test_traj, label=label, dblabel=dblabel
    )
    plter.plot_amp_error(
        energy_forcetrain,
        force_forcetrain,
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )
