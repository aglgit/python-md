import sys

sys.path.insert(0, "../tools")

import os
import numpy as np
import pandas as pd
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_rmses
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Cosine, Polynomial
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (3, 3, 3)
    temp = 500

    n_train = int(2e4)
    n_test = int(7.5e3)
    save_interval = 100

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
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

    Rcs = [4.0, 5.0, 6.0, 7.0, 8.0]
    cutoffs = []
    for Rc in Rcs:
        cutoffs.append(Cosine(Rc))
        cutoffs.append(Polynomial(Rc))

    calcs = {}
    for cutoff in cutoffs:
        trn = Trainer(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            activation=activation,
            cutoff=cutoff,
            Gs=Gs,
        )
        label = "{}-{}".format(cutoff.__class__.__name__, cutoff.Rc)
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
        columns = ["Cutoff", "Energy RMSE", "Force RMSE"]
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
