import sys

sys.path.insert(0, "../tools")

import os
from ase.io import read
from asap3 import EMT
from amp import Amp
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from plot import Plotter
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 300
    save_interval = 100

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")

    if not os.path.exists(train_traj):
        n_train = int(4e5)

        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_traj))
        ctrj.integrate_atoms(atoms, train_traj, n_train, save_interval)

    if not os.path.exists(test_traj):
        n_test = int(4e4)

        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(test_traj))
        ctrj.integrate_atoms(atoms, test_traj, n_test, save_interval)

    energy_coefficient = 1.0
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None
    max_steps = int(1e4)

    trn = Trainer(
        energy_coefficient=energy_coefficient,
        hidden_layers=hidden_layers,
        activation=activation,
        cutoff=cutoff,
        Gs=Gs,
    )

    if not os.path.exists("energy.amp"):
        energy_convergence = {
            "energy_rmse": 1e-9,
            "force_rmse": None,
            "max_steps": max_steps,
        }
        force_coefficient = None
        trn.convergence = energy_convergence
        trn.force_coefficient = force_coefficient
        amp_energy_calc = trn.create_calc(label="energy", dblabel="amp")
        print("Training from trajectory {}!".format(train_traj))
        try:
            amp_energy_calc.train(train_traj)
        except TrainingConvergenceError:
            amp_energy_calc.save("energy.amp", overwrite=True)
    else:
        amp_energy_calc = Amp.load("energy.amp")

    if not os.path.exists("force.amp"):
        force_convergence = {
            "energy_rmse": 1e-9,
            "force_rmse": 1e-3,
            "max_steps": max_steps,
        }
        force_coefficient = 0.05
        trn.convergence = force_convergence
        trn.force_coefficient = force_coefficient
        amp_force_calc = trn.create_calc(label="force", dblabel="amp")
        print("Training from trajectory {}!".format(train_traj))
        try:
            amp_force_calc.train(train_traj)
        except TrainingConvergenceError:
            amp_force_calc.save("force.amp", overwrite=True)
    else:
        amp_force_calc = Amp.load("force.amp")

    anl = Analyzer()
    plt = Plotter()
    plt_dir = "plots"

    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    energy_noforcetrain = os.path.join(plt_dir, "energy_noforcetrain.png")
    force_noforcetrain = os.path.join(plt_dir, "force_noforcetrain.png")
    energy_exact, energy_diff, force_exact, force_diff = anl.calculate_amp_error(
        test_traj, amp_energy_calc
    )
    plt.plot_amp_error(
        energy_noforcetrain,
        force_noforcetrain,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )

    energy_forcetrain = os.path.join(plt_dir, "energy_forcetrain.png")
    force_forcetrain = os.path.join(plt_dir, "force_forcetrain.png")
    energy_exact, energy_diff, force_exact, force_diff = anl.calculate_amp_error(
        test_traj, amp_force_calc
    )
    plt.plot_amp_error(
        energy_forcetrain,
        force_forcetrain,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )
