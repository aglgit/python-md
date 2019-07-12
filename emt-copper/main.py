import sys

sys.path.insert(0, "../tools")

import argparse
import os
from asap3 import EMT
from ase.io import read
from amp import Amp
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Polynomial
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from train_amp import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", dest="generate", action="store_true")
    parser.add_argument("--no-generate", dest="generate", action="store_false")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--no-train", dest="train", action="store_false")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.add_argument("--analyze", dest="analyze", action="store_true")
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.set_defaults(generate=False, train=False, test=False, analyze=False)

    args = parser.parse_args()
    generate = args.generate
    train = args.train
    test = args.test
    analyze = args.analyze

    system = "copper"
    size = (4, 4, 4)
    temp = 300

    metastep = 20
    train_dir = "trajs"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_trajs = [
        os.path.join(train_dir, "training_{}.traj".format(i))
        for i in range(1, metastep + 1)
    ]
    n_train = int(1e3)
    save_interval = 10

    convergence = {"energy_rmse": 1e-9, "force_rmse": 1e-3, "max_steps": int(1e3)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None

    n_test = int(1e3)
    test_traj = "test.traj"
    amp_test_traj = "amp_test.traj"

    if generate:
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        for i in range(metastep):
            train_traj = train_trajs[i]
            print("Creating trajectory {}".format(train_traj))
            ctrj.integrate_atoms(atoms, train_traj, n_train, save_interval)
        xyz_file = "training.xyz"
        ctrj.concat_trajectories(train_dir, xyz_file)

    if train:
        if os.path.exists("amp.amp"):
            print("Trained AMP calculator amp.amp already exists!")
        else:
            trn = Trainer(
                convergence=convergence,
                energy_coefficient=energy_coefficient,
                force_coefficient=force_coefficient,
                hidden_layers=hidden_layers,
                activation=activation,
                cutoff=cutoff,
                Gs=Gs,
            )
            amp_calc = trn.create_calc()

            print("Performing global search with annealer")
            images = read(train_trajs[0], ":")
            Annealer(calc=amp_calc, images=images, Tmax=20, Tmin=1, steps=500)

            for i in range(metastep):
                train_traj = train_trajs[i]
                print("Training from trajectory {}!".format(train_traj))
                amp_calc.train(train_traj)

    if test:
        if not os.path.exists("amp.amp"):
            print("No trained AMP calculator amp.amp exists!")
        else:
            atmb = AtomBuilder()
            atoms = atmb.build_atoms(system, size, temp)
            amp_atoms = atmb.build_atoms(system, size, temp)

            calc = EMT()
            atoms.set_calculator(calc)
            amp_calc = Amp.load("amp.amp")
            amp_atoms.set_calculator(amp_calc)

            ctrj = CreateTrajectory()
            ctrj.integrate_atoms(atoms, test_traj, n_test, save_interval)
            ctrj.steps = 0
            ctrj.integrate_atoms(amp_atoms, amp_test_traj, n_test, save_interval)

    if analyze:
        pass
