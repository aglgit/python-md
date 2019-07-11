import sys

sys.path.insert(0, "../tools")

import argparse
import os
import numpy as np
from ase.calculators.lj import LennardJones
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

    system = "argon"
    epsilon = 1.0318e-2
    sigma = 3.405
    rc = 2.5*sigma

    n_train = int(4e4)
    save_interval = 10
    size = (6, 6, 6)
    temp = 300

    train_traj = "training.traj"
    test_traj = "test.traj"
    amp_test_traj = "amp_test.traj"
    legend = ["Lennard-Jones", "AMP"]

    anl = Analyzer(save_interval)
    plt = Plotter()
    trn = Trainer()

    if generate:
        calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=rc)
        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(train_traj, n_train, save_interval)
        generator.convert_traj(train_traj)

    if train:
        convergence = {"energy_rmse": 1e-9, "force_rmse": None, "max_steps": int(2e3)}
        force_coefficient = None
        hidden_layers = (40, 20, 10)
        cutoff = Polynomial(5.0)

        elements = ["Ar"]
        Gs = make_symmetry_functions(
            elements=elements,
            type="G2",
            etas=np.logspace(np.log10(0.1), np.log10(5.0), num=8),
        )
        Gs += make_symmetry_functions(
            elements=elements,
            type="G5",
            etas=[0.005],
            zetas=[1.0, 2, 0, 4.0],
            gammas=[1.0, -1.0],
        )

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
            calc = LennardJones(epsilon=epsilon, sigma=sigma)
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
