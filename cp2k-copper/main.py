import os
import sys
import numpy as np
from ase.calculators.cp2k import CP2K
from amp import Amp
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import make_symmetry_functions
from amp.model import LossFunction

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 500

    n_train = int(8e2)
    n_train_force = int(5e2)
    n_test = int(2e2)
    save_interval = 1

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Cosine(4.0)

    elements = ["Cu"]
    nr = 4
    nz = 1
    gammas = [1.0, -1.0]
    radial_etas = 10.0 * np.ones(nr)
    centers = np.linspace(0.5, cutoff.Rc + 0.5, nr)
    angular_etas = np.linspace(0.1, 1.0, nr)
    zetas = [4 ** i for i in range(nz)]
    G2 = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )
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
    calc = CP2K()
    train_atoms = trjbd.build_atoms(system, size, temp, calc)
    calc = CP2K()
    test_atoms = trjbd.build_atoms(system, size, temp, calc)
    CP2K.command = "env OMP_NUM_THREADS=16 mpiexec -np 4 cp2k_shell.psmp"

    train_traj = "training.traj"
    train_force_traj = "training_force.traj"
    test_traj = "test.traj"
    steps, train_traj = trjbd.integrate_atoms(
        train_atoms, train_traj, n_train, save_interval
    )
    steps, train_force_traj = trjbd.integrate_atoms(
        train_atoms, train_force_traj, n_train_force, save_interval, steps=steps
    )
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval
    )

    label = "energy-trained"
    calc = trn.create_calc(label=label, dblabel=label)
    ann = Annealer(calc=calc, images=train_traj, Tmax=20, Tmin=1, train_forces=False)
    amp_name = trn.train_calc(calc, train_traj)

    label = os.path.join("calcs", "force-trained")
    calc = Amp.load(amp_name, label=label, dblabel=label)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    loss_function = LossFunction(
        convergence=convergence, energy_coefficient=1.0, force_coefficient=0.1
    )
    calc.model.lossfunction = loss_function
    amp_name = trn.train_calc(calc, train_traj)
