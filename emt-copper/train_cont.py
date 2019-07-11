import sys

sys.path.insert(0, "../tools")

import argparse
import os
import numpy as np
from asap3 import EMT
from amp import Amp
from amp.descriptor.cutoffs import Polynomial
from amp.descriptor.gaussian import make_symmetry_functions
from analysis import Analyzer
from generate_traj import GenerateTrajectory
from plot import Plotter
from trainer import Trainer


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

    n_train = int(6e4)
    save_interval = 10
    size = (4, 4, 4)
    temp = 2000

    train_traj = "training.traj"
    test_traj = "test.traj"
    amp_test_traj = "amp_test.traj"
    legend = ["EMT", "AMP"]

    anl = Analyzer(save_interval)
    plt = Plotter()
    trn = Trainer()

    if train_cont:
        train_folder = "traj"
        if not os.path.exists(train_folder)
            os.mkdir(train_folder)
        train_trajs = [os.path.join(train_folder, "training_{}.traj".format(i)) for i in range(1,11)]
        calc = EMT()
        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(train_traj, n_train, save_interval)
        generator.convert_traj(train_traj)


    if generate:
        calc = EMT()
        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(train_traj, n_train, save_interval)

        convergence = {"energy_rmse": 1e-9, "force_rmse": None, "max_steps": int(2e3)}
        force_coefficient = None
        hidden_layers = (40, 40)
        cutoff = Polynomial(6.0)
        Gs = None

        trn.train_amp(
            train_traj,
            convergence=convergence,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            cutoff=cutoff,
            Gs=Gs,
        )

    if train:
        convergence = {"energy_rmse": 1e-9, "force_rmse": None, "max_steps": int(2e3)}
        force_coefficient = None
        hidden_layers = (40, 40)
        cutoff = Polynomial(6.0)
        Gs = None

        if not os.path.exists("amp.amp"):
            trn.train_amp(
                train_traj,
                convergence=convergence,
                force_coefficient=force_coefficient,
                hidden_layers=hidden_layers,
                cutoff=cutoff,
                Gs=Gs,
            )
        else:
            print("Trained AMP calculator amp.amp already exists!")

    if test:
        if os.path.exists("amp.amp"):
            calc = EMT()
            amp_calc = Amp.load("amp.amp")
            trn.test_amp(
                calc, system, amp_calc, test_traj, amp_test_traj, size=size, temp=temp
            )
        else:
            print("No trained AMP calculator amp.amp!") 
    if analyze:
        steps, exact_energy, amp_energy = anl.calculate_energy_diff(
            test_traj, amp_test_traj
        )
        plt.plot_energy_diff("energy.png", legend, steps, exact_energy, amp_energy)
