import sys

sys.path.insert(1, "../tools")

import numpy as np
import pandas as pd
from asap3 import EMT
from amp.analysis import calculate_rmses
from amp.descriptor.cutoffs import Cosine, Polynomial
from create_trajectory import TrajectoryBuilder
from trainer import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 500
    seed = 0

    n_train = int(8e2)
    n_test = int(2e2)
    save_interval = 100

    trjbd = TrajectoryBuilder()
    calc = EMT()
    train_atoms = trjbd.build_atoms(system, size, temp, calc)
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc)

    train_traj = "training.traj"
    test_traj = "test.traj"
    train_steps, train_traj = trjbd.integrate_atoms(
        train_atoms, train_traj, n_train, save_interval, convert=True
    )
    test_steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval, convert=True
    )

    max_steps = int(2e1)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    Gs = None
    trn = Trainer(
        convergence=convergence,
        force_coefficient=force_coefficient,
        hidden_layers=hidden_layers,
        activation=activation,
        Gs=Gs,
    )

    Rcs = [4.0, 5.0, 6.0, 7.0, 8.0]
    cutoffs = []
    for Rc in Rcs:
        cutoffs.append(Cosine(Rc))
        cutoffs.append(Polynomial(Rc))

    calcs = {}
    for cutoff in cutoffs:
        trn.cutoff = cutoff
        label = "{}-{}".format(cutoff.__class__.__name__, cutoff.Rc)
        dblabel = "train-" + label
        calc = trn.create_calc(label=label, dblabel=dblabel)
        amp_name = trn.train_calc(calc, train_traj)

        calcs[label] = amp_name

    logfile = "log.txt"
    if not os.path.exists(logfile):
        columns = ["Cutoff", "Energy RMSE", "Force RMSE"]
        df = pd.DataFrame(columns=columns)

        for i, (label, amp_name) in enumerate(calcs.items()):
            print("Testing calculator {} on trajectory {}".format(amp_name, test_traj))
            dblabel = "test-" + label
            energy_rmse, force_rmse = calculate_rmses(
                amp_name, test_traj, dblabel=dblabel
            )

            row = [label, energy_rmse, force_rmse]
            df.loc[i] = row
            df.to_csv(logfile, index=False)

    df = pd.read_csv(
        logfile, dtype={"Energy RMSE": np.float64, "Force RMSE": np.float64}
    )
    print(df.to_latex(float_format="{:.2E}".format, index=False))
