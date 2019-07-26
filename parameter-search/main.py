import sys

sys.path.insert(0, "../tools")

import os
import pandas as pd
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_rmses
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
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

    parameters = {}
    parameters["force_coefficient"] = [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    parameters["hidden_layers"] = [
        [10],
        [20],
        [30],
        [40],
        [10, 10],
        [20, 10],
        [20, 20],
        [30, 10],
        [40, 40],
    ]
    parameters["activation"] = ["tanh", "sigmoid"]
    parameters["energy_rmse"] = []
    parameters["force_rmse"] = []

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")

    calc_dir = "calcs"
    if not os.path.exists(calc_dir):
        os.mkdir(calc_dir)
    dblabel = "amp"

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

    calcs = []
    for ac in parameters["activation"]:
        for hl in parameters["hidden_layers"]:
            for fc in parameters["force_coefficient"]:
                trn = Trainer(
                    convergence=convergence,
                    energy_coefficient=energy_coefficient,
                    force_coefficient=fc,
                    hidden_layers=hl,
                    activation=ac,
                    cutoff=cutoff,
                    Gs=Gs,
                )
                label = "{}-{}-{}".format(ac, hl, fc)
                label = os.path.join(calc_dir, label)
                amp_name = "{}.amp".format(label)
                if not os.path.exists(amp_name):
                    print(label)
                    amp_calc = trn.create_calc(label=label, dblabel=dblabel)
                    try:
                        amp_calc.train(train_traj)
                    except TrainingConvergenceError:
                        amp_calc.save(amp_name, overwrite=True)
                calcs.append(amp_name)

    if not os.path.exists(logfile):
        columns = ["Activation", "Hidden layers", "Force coefficient", "Energy RMSE", "Force RMSE"]
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(logfile)

    for i, calc in enumerate(calcs):
        print(calc)
        ac, hl, fc = calc.split("/")[-1].split(".amp")[0].split("-")
        print(ac, hl, fc)
        data = calculate_rmses(calc, test_traj, dblabel=dblabel)

        row = [ac, hl, fc, data["energy_rmse"], data["force_rmse"]]
        df.loc[i] = row
        df.to_csv(logfile, sep=" ")
