import sys

sys.path.insert(0, "../tools")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from ase.io import read
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
    size = (2, 2, 2)
    temp = 500

    n_train = [10, 20, 50, 100, 200, 500, 1000, 2000]
    n_test = int(5e3)
    save_interval = 100

    max_steps = int(5e2)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    force_coefficient = 0.1

    logfile = "log.txt"

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_trajs = []
    test_traj = os.path.join(traj_dir, "test.traj")

    calc_dir = "calcs"
    if not os.path.exists(calc_dir):
        os.mkdir(calc_dir)

    for nt in n_train:
        train_traj = os.path.join(traj_dir, "train_n{}.traj".format(nt))
        train_trajs.append(train_traj)
        if not os.path.exists(train_traj):
            atmb = AtomBuilder()
            atoms = atmb.build_atoms(system, size, temp)
            calc = EMT()
            atoms.set_calculator(calc)

            ctrj = CreateTrajectory()
            print("Creating trajectory {}".format(train_traj))
            ctrj.integrate_atoms(atoms, train_traj, nt * save_interval, save_interval)
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

    calcs = {}
    for i, train_traj in enumerate(train_trajs):
        trn = Trainer(convergence=convergence, force_coefficient=force_coefficient)
        label = "n{}".format(n_train[i])
        amp_label = os.path.join(calc_dir, label)
        amp_dblabel = amp_label + "-train"
        amp_name = amp_label + ".amp"
        if not os.path.exists(amp_name):
            print("Training {}".format(amp_name))
            amp_calc = trn.create_calc(label=amp_label, dblabel=amp_dblabel)
            try:
                amp_calc.train(train_traj)
            except TrainingConvergenceError:
                amp_calc.save(amp_name, overwrite=True)

        calcs[label] = amp_name

    if not os.path.exists(logfile):
        columns = ["Number of data points", "Energy RMSE", "Force RMSE"]
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
