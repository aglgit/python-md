import os
import numpy as np
import pandas as pd
from amp import Amp
from amp.utilities import TrainingConvergenceError, Annealer
from amp.analysis import calculate_rmses
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Trainer:
    def __init__(
        self,
        convergence=None,
        energy_coefficient=1.0,
        force_coefficient=None,
        hidden_layers=(10, 10),
        activation="tanh",
        cutoff=6.0,
        Gs=None,
    ):
        if convergence is None:
            self.convergence = {
                "energy_rmse": 1e-3,
                "force_rmse": None,
                "max_steps": int(1e3),
            }
        else:
            self.convergence = convergence

        self.energy_coefficient = energy_coefficient
        self.force_coefficient = force_coefficient
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.cutoff = cutoff
        self.Gs = Gs

    def create_calc(self, label="amp", dblabel="amp", calc_dir="calcs"):
        if not os.path.exists(calc_dir):
            os.mkdir(calc_dir)
        amp_label = os.path.join(calc_dir, label)
        amp_dblabel = os.path.join(calc_dir, dblabel)
        amp_name = amp_label + ".amp"
        if not os.path.exists(amp_name):
            print("Creating calculator {}...".format(amp_name))
            loss_function = LossFunction(
                convergence=self.convergence,
                energy_coefficient=self.energy_coefficient,
                force_coefficient=self.force_coefficient,
            )
            model = NeuralNetwork(
                hiddenlayers=self.hidden_layers,
                activation=self.activation,
                lossfunction=loss_function,
            )
            descriptor = Gaussian(cutoff=self.cutoff, Gs=self.Gs, fortran=True)
            calc = Amp(
                descriptor=descriptor, model=model, label=amp_label, dblabel=amp_dblabel
            )

            return calc
        else:
            print("Calculator {} already exists!".format(amp_name))
            calc = Amp.load(amp_name, label=amp_label, dblabel=amp_dblabel)

            return calc

    def train_calc(self, calc, traj_file, calc_dir="calcs", traj_dir="trajs"):
        label = calc.label
        amp_name = label + ".amp"
        if not os.path.exists(amp_name):
            print(
                "Training calculator {} from trajectory {}...".format(
                    amp_name, traj_file
                )
            )
            try:
                calc.train(traj_file)
            except TrainingConvergenceError:
                calc.save(amp_name, overwrite=True)

            return amp_name
        else:
            print("Trained calculator {} already exists!".format(amp_name))

            return amp_name

    def train_calculators(self, parameter, parameter_dict, traj_file, dblabel=None):
        calcs = {}
        for label, param in parameter_dict.items():
            setattr(self, parameter, param)
            if dblabel is None:
                dblabel = label
            calc = self.create_calc(label=label, dblabel=dblabel)
            ann = Annealer(
                calc=calc,
                images=traj_file,
                Tmax=20,
                Tmin=1,
                steps=4000,
                train_forces=False,
            )
            amp_name = self.train_calc(calc, traj_file)
            calcs[label] = amp_name

        return calcs

    def test_calculators(
        self,
        calcs,
        traj_file,
        columns,
        logfile="log.txt",
        calc_dir="calcs",
        dblabel=None,
    ):
        if not os.path.exists(logfile):
            df = pd.DataFrame(columns=columns)
            for i, (label, amp_name) in enumerate(calcs.items()):
                print(
                    "Testing calculator {} on trajectory {}...".format(
                        amp_name, traj_file
                    )
                )
                if dblabel is None:
                    dblabel = os.path.join(calc_dir, label + "-test")
                energy_rmse, force_rmse = calculate_rmses(
                    amp_name, traj_file, dblabel=dblabel
                )

                row = [label, energy_rmse, force_rmse]
                df.loc[i] = row
                df.to_csv(logfile, index=False)
        else:
            print("Logfile {} already exists!".format(logfile))

        df = pd.read_csv(
            logfile, dtype={"Energy RMSE": np.float64, "Force RMSE": np.float}
        )
        print(df.to_latex(float_format="{:.2E}".format, index=False))
