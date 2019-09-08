import os
import numpy as np
import pandas as pd
from amp import Amp
from amp.utilities import TrainingConvergenceError
from amp.analysis import calculate_rmses
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Trainer:
    def __init__(
        self,
        convergence=None,
        energy_coefficient=1.0,
        force_coefficient=None,
        overfit=1e-3,
        hidden_layers=(10, 10),
        activation="tanh",
        cutoff=Cosine(6.0),
        Gs=None,
        calc_dir="calcs",
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
        self.overfit = overfit
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.cutoff = cutoff
        self.Gs = Gs

        if not os.path.exists(calc_dir):
            os.mkdir(calc_dir)
        self.calc_dir = calc_dir

    def create_Gs(
        self, elements, num_radial_etas, num_angular_etas, num_zetas, angular_type
    ):
        uncentered_etas = np.linspace(1.0, 20.0, num_radial_etas)
        centers = np.zeros(num_radial_etas)
        G2_uncentered = make_symmetry_functions(
            elements=elements, type="G2", etas=uncentered_etas, centers=centers
        )

        centered_etas = 5.0 * np.ones(num_radial_etas)
        centers = np.linspace(0.5, self.cutoff.Rc - 0.5, num_radial_etas)
        G2_centered = make_symmetry_functions(
            elements=elements, type="G2", etas=centered_etas, centers=centers
        )

        angular_etas = np.linspace(0.01, 3.0, num_angular_etas)
        zetas = [2 ** i for i in range(num_zetas)]
        G_ang = make_symmetry_functions(
            elements=elements,
            type=angular_type,
            etas=angular_etas,
            zetas=zetas,
            gammas=[1.0, -1.0],
        )

        self.Gs = G2_uncentered + G2_centered + G_ang

    def create_calc(self, label, dblabel):
        amp_label = os.path.join(self.calc_dir, label)
        amp_dblabel = os.path.join(self.calc_dir, dblabel)
        amp_name = amp_label + ".amp"
        if not os.path.exists(amp_name):
            print("Creating calculator {}...".format(amp_name))
            loss_function = LossFunction(
                convergence=self.convergence,
                energy_coefficient=self.energy_coefficient,
                force_coefficient=self.force_coefficient,
                overfit=self.overfit,
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

    def train_calc(self, calc, traj_file):
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

    def test_calculators(
        self, calcs, traj_file, columns, logfile="log.txt", dblabel=None
    ):
        if not os.path.exists(logfile):
            df = pd.DataFrame(columns=columns)
            for i, (label, amp_name) in enumerate(calcs.items()):
                print(
                    "Testing calculator {} on trajectory {}...".format(
                        amp_name, traj_file
                    )
                )
                amp_label = os.path.join(self.calc_dir, label)
                if dblabel is None:
                    amp_dblabel = amp_label + "-test"
                else:
                    amp_dblabel = os.path.join(self.calc_dir, dblabel)
                energy_rmse, force_rmse = calculate_rmses(
                    amp_name, traj_file, label=amp_label, dblabel=amp_dblabel
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
