import sys

sys.path.insert(0, "../tools")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from copy import deepcopy
from ase.io import read
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_rmses, calculate_energy_diff
from amp.utilities import TrainingConvergenceError
from amp.descriptor.cutoffs import Polynomial
from amp.descriptor.gaussian import make_symmetry_functions, Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from analysis import Analyzer
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from plot import Plotter
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 500

    n_train = int(1e4)
    n_test = int(5e3)
    save_interval = 100

    max_steps = int(2e1)

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

    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None

    descriptor = Gaussian(cutoff=cutoff, Gs=Gs, fortran=True)
    loss_function = LossFunction(
        convergence=convergence, force_coefficient=force_coefficient
    )
    model = NeuralNetwork(
        hiddenlayers=hidden_layers, activation=activation, lossfunction=loss_function
    )
    calc = Amp(descriptor=descriptor, model=model)
    calc.train(train_traj)
    calc.save("test.amp")

    amp_calc_unsampled = Amp.load("test.amp")
    amp_calc_sampled = Amp.load("test.amp")

    train_trajs = [
        os.path.join(calc_dir, "train_{}.traj".format(traj))
        for traj in ["unsampled", "sampled"]
    ]
    atmb = AtomBuilder()
    atoms = atmb.build_atoms(system, size, temp)

    atoms_unsampled = atoms
    calc = EMT()
    atoms_unsampled.set_calculator(calc)
    calc = EMT()
    atoms_sampled.set_calculator(calc)

    n_images = 500
    if not os.path.exists(train_trajs[0]):
        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_trajs[0]))
        ctrj.integrate_atoms(
            atoms_unsampled, train_trajs[0], n_images * save_interval, save_interval
        )
        ctrj.convert_trajectory(train_traj)

    amp_calc_unsampled.train(train_trajs[0])

    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    loss_function = LossFunction(
        convergence=convergence, force_coefficient=force_coefficient
    )
    amp_calc_sampled.model.lossfunction = loss_function

    n_images = 5000
    if not os.path.exists(train_trajs[1]):
        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_trajs[1]))
        ctrj.integrate_atoms(
            atoms_sampled, train_trajs[1], n_images * save_interval, save_interval
        )
        ctrj.convert_trajectory(train_traj)

    amp_calc_sampled.save("test.amp")
    energy_rmse, energy_exact, energy_diff, images = calculate_energy_diff(
        "test.amp", train_trajs[1]
    )
