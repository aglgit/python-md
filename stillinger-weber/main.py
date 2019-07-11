import sys

sys.path.insert(0, "../tools")

import argparse
import os
from asap3 import OpenKIMcalculator
from amp import Amp
from generate_traj import GenerateTrajectory
from trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", dest="generate", action="store_true")
    parser.add_argument("--no-generate", dest="generate", action="store_false")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--no-train", dest="train", action="store_false")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.set_defaults(generate=False, train=False, test=False)

    args = parser.parse_args()
    generate = args.generate
    train = args.train
    test = args.test

    system = "silicon"
    train_traj = "training.traj"

    n_train = int(1e5)
    save_interval = 10
    size = (6, 6, 6)
    temp = 300

    trn = Trainer()

    if generate:
        calc = OpenKIMcalculator("SW_StillingerWeber_1985_Si__MO_405512056662_005")
        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(train_traj, n_train, save_interval)
        generator.convert_traj(train_traj)

    if train:
        train_traj = "training.traj"
        convergence = {"energy_rmse": 1e-9, "force_rmse": None, "max_steps": int(2e3)}
        force_coefficient = None
        hidden_layers = (40, 20, 10)
        cutoff = 6.0
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
        test_traj = "test.traj"
        amp_test_traj = "amp_test.traj"

        if os.path.exists("amp.amp"):
            calc = OpenKIMcalculator("SW_StillingerWeber_1985_Si__MO_405512056662_005")
            amp_calc = Amp.load("amp.amp")
            trn.test_amp(
                calc, system, amp_calc, test_traj, amp_test_traj, size=size, temp=temp
            )
        else:
            print("No trained AMP calculator amp.amp!")
