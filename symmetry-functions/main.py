import sys

sys.path.insert(0, "../tools")

import os
import numpy as np
import pandas as pd
from asap3 import EMT
from amp import Amp
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from amp.descriptor.gaussian import make_symmetry_functions
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 300

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    test_traj = os.path.join(traj_dir, "test.traj")
    logfile = "log.txt"

    n_train = int(5e2)
    n_test = int(5e2)
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

    # Defaults
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": int(1e1)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)

    elements = ["Cu"]
    etas_radial = [6, 7, 8, 9, 10]
    etas_angular = [2, 2, 3, 3, 3, 3]
    zetas = [1.0, 2.0, 4.0, 16.0]
    gammas = [1.0, -1.0]
    Gs = [None]
    for etar, etaa in zip(etas_radial, etas_angular):
        G2 = make_symmetry_functions(
            elements=elements,
            type="G2",
            etas=np.logspace(np.log10(0.01), np.log10(5.0), num=etar),
        )
        G4 = make_symmetry_functions(
            elements=elements,
            type="G4",
            etas=np.logspace(np.log10(0.001), np.log10(0.1), num=etaa),
            zetas=zetas,
            gammas=gammas,
        )
        G5 = make_symmetry_functions(
            elements=elements,
            type="G5",
            etas=np.logspace(np.log10(0.001), np.log10(0.1), num=etaa),
            zetas=zetas,
            gammas=gammas,
        )

        Gs.append(G2 + G4)
        Gs.append(G2 + G5)

    anl = Analyzer(save_interval=save_interval)

    results = {
        "Num radial": [],
        "Num angular": [],
        "Angular type": [],
        "Energy RMSE": [],
        "Force RMSE": [],
    }
    for i in range(len(etas_radial)):
        for j, Gtype in enumerate(["G4", "G5"]):
            G = Gs[2 * i + j]
            num_radial = etas_radial[i]
            num_angular = etas_angular[i] * len(zetas) * len(gammas)

            trn = Trainer(
                convergence=convergence,
                energy_coefficient=energy_coefficient,
                force_coefficient=force_coefficient,
                hidden_layers=hidden_layers,
                activation=activation,
                cutoff=cutoff,
                Gs=G,
            )
            label = "Gs-{}-{}-{}".format(num_radial, num_angular, Gtype)
            print(label)
            amp_calc = trn.create_calc(label=label, dblabel=label)

            try:
                amp_calc.train(train_traj)
            except TrainingConvergenceError:
                pass

            energy_rmse, force_rmse = anl.calculate_rmses(test_traj, amp_calc)

            results["Num radial"].append(num_radial)
            results["Num angular"].append(num_angular)
            results["Angular type"].append(Gtype)
            results["Energy RMSE"].append(energy_rmse)
            results["Force RMSE"].append(force_rmse)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(logfile, sep=" ")
