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
    size = (2, 2, 2)
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
    max_steps = int(1e3)

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

    exit(1)

    # Defaults
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)

    elements = ["Cu"]
    num_etas_radial = [4, 6, 7, 8, 9, 10]
    num_etas_angular = [1, 2, 2, 3, 3, 3]
    num_zetas = [2, 2, 2, 3, 3, 4]
    gammas = [1.0, -1.0]
    Gs = {"Default": None}
    for netar, netaa, nzeta in zip(num_etas_radial, num_etas_angular, num_zetas):
        G2 = make_symmetry_functions(
            elements=elements, type="G2", etas=np.linspace(0.1, 8.0, num=netar)
        )
        G4 = make_symmetry_functions(
            elements=elements,
            type="G4",
            etas=np.linspace(0.01, 1.0, num=netaa),
            zetas=[2 ** i for i in range(nzeta)],
            gammas=gammas,
        )
        G5 = make_symmetry_functions(
            elements=elements,
            type="G5",
            etas=np.linspace(0.01, 1.0, num=netaa),
            zetas=[2 ** i for i in range(nzeta)],
            gammas=gammas,
        )

        num_radial = netar
        num_angular = netaa * nzeta * len(gammas)
        key_G4 = "Gs-{}-{}-{}".format(num_radial, num_angular, "G4")
        key_G5 = "Gs-{}-{}-{}".format(num_radial, num_angular, "G5")

        Gs[key_G4] = G2 + G4
        Gs[key_G5] = G2 + G5

    anl = Analyzer(save_interval=save_interval)

    results = {"Key": [], "Energy RMSE": [], "Force RMSE": []}

    for key in Gs:
        print(key)
        G = Gs[key]
        trn = Trainer(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            activation=activation,
            cutoff=cutoff,
            Gs=G,
        )
        amp_calc = trn.create_calc(label=key, dblabel=key)

        try:
            amp_calc.train(train_traj)
        except TrainingConvergenceError:
            pass

        energy_rmse, force_rmse = anl.calculate_rmses(test_traj, amp_calc)

        results["Key"].append(key)
        results["Energy RMSE"].append(energy_rmse)
        results["Force RMSE"].append(force_rmse)

        df = pd.DataFrame.from_dict(results)
        df.to_csv(logfile, sep=" ")
