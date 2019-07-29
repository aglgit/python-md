import sys

sys.path.insert(0, "../tools")

import os
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_error
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from plot import Plotter
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 500

    n_train = int(8e2)
    n_test = int(2e2)
    save_interval = 100

    energy_coefficient = 1.0
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None
    max_steps = int(1e1)

    trn = Trainer(
        energy_coefficient=energy_coefficient,
        hidden_layers=hidden_layers,
        activation=activation,
        cutoff=cutoff,
        Gs=Gs,
    )

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")

    calc_dir = "calcs"
    if not os.path.exists(calc_dir):
        os.mkdir(calc_dir)
    energy_calc = os.path.join(calc_dir, "energy.amp")
    energy_label = os.path.join(calc_dir, "energy")
    force_calc = os.path.join(calc_dir, "force.amp")
    force_label = os.path.join(calc_dir, "force")

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    energy_noforcetrain = os.path.join(plot_dir, "energy_noforcetrain.png")
    force_noforcetrain = os.path.join(plot_dir, "force_noforcetrain.png")
    energy_forcetrain = os.path.join(plot_dir, "energy_forcetrain.png")
    force_forcetrain = os.path.join(plot_dir, "force_forcetrain.png")

    if not os.path.exists(train_traj):
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_traj))
        ctrj.integrate_atoms(atoms, train_traj, n_train, save_interval)
        ctrj.convert_trajectory(train_traj)

    if not os.path.exists(test_traj):
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(test_traj))
        ctrj.integrate_atoms(atoms, test_traj, n_test, save_interval)
        ctrj.convert_trajectory(test_traj)

    dblabel = "amp-train"
    if not os.path.exists(energy_calc):
        energy_convergence = {
            "energy_rmse": 1e-16,
            "force_rmse": None,
            "max_steps": max_steps,
        }
        force_coefficient = None
        trn.convergence = energy_convergence
        trn.force_coefficient = force_coefficient
        amp_energy_calc = trn.create_calc(label=energy_label, dblabel=dblabel)

        print("Training from trajectory {}!".format(train_traj))
        try:
            amp_energy_calc.train(train_traj)
        except TrainingConvergenceError:
            amp_energy_calc.save(energy_calc, overwrite=True)

    if not os.path.exists(force_calc):
        force_convergence = {
            "energy_rmse": 1e-16,
            "force_rmse": 1e-16,
            "max_steps": max_steps,
        }
        force_coefficient = 0.1
        trn.convergence = force_convergence
        trn.force_coefficient = force_coefficient
        amp_force_calc = trn.create_calc(label=force_label, dblabel=dblabel)

        print("Training from trajectory {}!".format(train_traj))
        try:
            amp_force_calc.train(train_traj)
        except TrainingConvergenceError:
            amp_force_calc.save(force_calc, overwrite=True)

    dblabel = "amp-test"
    if not os.path.exists(energy_noforcetrain):
        plter = Plotter()

        energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
            energy_calc, images=test_traj, dblabel=dblabel
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

    if not os.path.exists(energy_forcetrain):
        energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
            force_calc, images=test_traj, dblabel=dblabel
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
