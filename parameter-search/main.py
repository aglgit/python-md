import sys

sys.path.insert(0, "../tools")

import os
import pandas as pd
from asap3 import EMT
from amp import Amp
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 300

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")
    logfile = "log.txt"

    n_train = int(5e4)
    n_test = int(7.5e3)
    save_interval = 100

    if not os.path.exists(train_traj):
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_traj))
        ctrj.integrate_atoms(atoms, train_traj, n_train, save_interval)

    if not os.path.exists(test_traj):
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(test_traj))
        ctrj.integrate_atoms(atoms, test_traj, n_test, save_interval)

    parameters = {}
    parameters["force_coefficient"] = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    parameters["hidden_layers"] = [
        [10],
        [20],
        [30],
        [40],
        [10, 10],
        [20, 10],
        [30, 10],
        [20, 20],
        [40, 40],
    ]
    parameters["activation"] = ["tanh", "sigmoid"]
    parameters["energy_rmse"] = []
    parameters["force_rmse"] = []

    # Defaults
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": int(1e3)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None

    anl = Analyzer(save_interval=save_interval)

    results = {
        "Activation": [],
        "Hidden layers": [],
        "Force coefficient": [],
        "Energy RMSE": [],
        "Force RMSE": [],
    }
    for ac in parameters["activation"]:
        for hl in parameters["hidden_layers"]:
            for fc in parameters["force_coefficient"]:
                print(ac, hl, fc)
                trn = Trainer(
                    convergence=convergence,
                    energy_coefficient=energy_coefficient,
                    force_coefficient=fc,
                    hidden_layers=hl,
                    activation=ac,
                    cutoff=cutoff,
                    Gs=Gs,
                )
                amp_calc = trn.create_calc()
                try:
                    amp_calc.train(train_traj)
                except TrainingConvergenceError:
                    pass

                energy_rmse, force_rmse = anl.calculate_rmses(test_traj, amp_calc)

                results["Activation"].append(ac)
                results["Hidden layers"].append(hl)
                results["Force coefficient"].append(fc)
                results["Energy RMSE"].append(energy_rmse)
                results["Force RMSE"].append(force_rmse)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(logfile, sep=" ")
