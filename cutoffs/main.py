import sys

sys.path.insert(0, "../tools")

import os
import pandas as pd
from asap3 import EMT
from amp import Amp
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

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")
    logfile = "log.txt"

    n_train = int(5e4)
    n_test = int(1e4)
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

    rcs = [4.0, 5.0, 6.0, 7.0, 8.0]
    cutoffs = []
    for rc in rcs:
        cutoffs.append(Cosine(rc))
        cutoffs.append(Polynomial(rc))

    # Defaults
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": int(1e3)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    Gs = None

    anl = Analyzer(save_interval=save_interval)

    results = {"Cutoff": [], "Energy RMSE": [], "Force RMSE": []}

    for cutoff in cutoffs:
        print(cutoff)
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
        amp_calc = trn.create_calc(label=label, dblabel=label)

        try:
            amp_calc.train(train_traj)
        except TrainingConvergenceError:
            pass

        energy_rmse, force_rmse = anl.calculate_rmses(test_traj, amp_calc)

        results["Cutoff"].append(label)
        results["Energy RMSE"].append(energy_rmse)
        results["Force RMSE"].append(force_rmse)

        df = pd.DataFrame.from_dict(results)
        df.to_csv(logfile, sep=" ")
