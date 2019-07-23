import sys

sys.path.insert(0, "../tools")

import argparse
import os
from asap3 import EMT
from ase.io import read
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.tflow2 import NeuralNetwork
from amp.descriptor.cutoffs import Polynomial
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from train_amp import Trainer

if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 300

    train_dir = "trajs"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_traj = os.path.join(train_dir, "training.traj")
    n_train = int(5e3)
    save_interval = 1000

    convergence = {"energy_rmse": 1e-9, "force_rmse": 1e-3, "max_steps": int(1e3)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None

    if not os.path.exists(train_traj):
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_traj))
        ctrj.integrate_atoms(atoms, train_traj, n_train, save_interval)

    descriptor = Gaussian(cutoff=cutoff, Gs=Gs, fortran=True)
    model = NeuralNetwork()
    calc = Amp(descriptor=descriptor, model=model)
    calc.train(train_traj)
