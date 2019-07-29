import sys

sys.path.insert(0, "../tools")

import os
import numpy as np
import pandas as pd
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_rmses
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from amp.descriptor.gaussian import make_symmetry_functions
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

    max_steps = int(1e1)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None

    logfile = "log.txt"

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")

    calc_dir = "calcs"
    if not os.path.exists(calc_dir):
        os.mkdir(calc_dir)

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

    elements = ["Cu"]
    num_radial = [5, 6, 7, 8, 9, 10]
    num_zetas = [1, 1, 1, 1, 2, 3]
    gammas = [1.0, -1.0]
    symm_funcs = {}
    for i in range(len(num_radial)):
        for shift in ["uncentered", "centered"]:
            nr = num_radial[i]
            nz = num_zetas[i]

            if shift == "uncentered":
                radial_grid = np.linspace(1.0, cutoff.Rc - 0.5, nr)
                etas = 1.0 / ((2 * radial_grid) ** 2)
                centers = np.zeros(len(etas))
            elif shift == "centered":
                radial_grid = np.linspace(0.5, cutoff.Rc - 0.5, nr)
                delta_r = (radial_grid[-1] - radial_grid[0]) / (nr - 1)
                etas = 1.0 / ((2 * delta_r) ** 2) * np.ones(nr)
                centers = radial_grid

            G2 = make_symmetry_functions(
                elements=elements, type="G2", etas=etas, centers=centers
            )

            G4 = make_symmetry_functions(
                elements=elements,
                type="G4",
                etas=etas,
                zetas=[2 ** i for i in range(nz)],
                gammas=[1.0, -1.0],
            )

            G5 = make_symmetry_functions(
                elements=elements,
                type="G5",
                etas=etas,
                zetas=[2 ** i for i in range(nz)],
                gammas=[1.0, -1.0],
            )

            label_G4 = "Gs-{}-{}-{}-{}".format(nr, nz, "G4", shift)
            label_G5 = "Gs-{}-{}-{}-{}".format(nr, nz, "G5", shift)

            symm_funcs[label_G4] = G2 + G4
            symm_funcs[label_G5] = G2 + G5

    calcs = {}
    for i, (label, symm_func) in enumerate(symm_funcs.items()):
        trn = Trainer(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            activation=activation,
            cutoff=cutoff,
            Gs=symm_func,
        )
        amp_label = os.path.join(calc_dir, label)
        amp_dblabel = amp_label + "-train"
        amp_name = amp_label + ".amp"
        if not os.path.exists(amp_name):
            print("Training {}".format(label))
            amp_calc = trn.create_calc(label=amp_label, dblabel=amp_dblabel)
            try:
                amp_calc.train(train_traj)
            except TrainingConvergenceError:
                amp_calc.save(amp_name, overwrite=True)

        calcs[label] = amp_name

    if not os.path.exists(logfile):
        columns = ["Symmetry functions", "Energy RMSE", "Force RMSE"]
        df = pd.DataFrame(columns=columns)

        for i, (label, amp_name) in enumerate(calcs.items()):
            print("Testing {}".format(amp_name))
            amp_dblabel = os.path.join(calc_dir, label) + "-test"
            energy_rmse, force_rmse = calculate_rmses(
                amp_name, test_traj, dblabel=amp_dblabel
            )

            row = [label, energy_rmse, force_rmse]
            df.loc[i] = row
            df.to_csv(logfile, index=False)

    df = pd.read_csv(
        logfile, dtype={"Energy RMSE": np.float64, "Force RMSE": np.float64}
    )
    print(df.to_latex(float_format="{:.2E}".format, index=False))
