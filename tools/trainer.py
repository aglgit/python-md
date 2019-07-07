import os
import ase.io
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from generate_traj import GenerateTrajectory


class Trainer:
    def train_amp(
        self,
        traj_file,
        convergence=None,
        energy_coefficient=1.0,
        force_coefficient=None,
        hidden_layers=(10),
        activation="tanh",
        cutoff=6.0,
        Gs=None,
    ):
        print("Training from traj: {}".format(traj_file))
        traj = ase.io.read(traj_file, ":")

        if convergence is None:
            convergence = {"energy_rmse": 1e-3, "max_steps": int(1e3)}

        descriptor = Gaussian(cutoff=cutoff, Gs=Gs, fortran=True)
        loss_function = LossFunction(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
        )
        model = NeuralNetwork(
            hiddenlayers=hidden_layers,
            activation=activation,
            lossfunction=loss_function,
        )

        if os.path.exists("amp.amp"):
            calc = Amp.load("amp.amp")
            calc.train(images=traj)
        else:
            calc = Amp(descriptor=descriptor, model=model)
            calc.train(images=traj)

    def test_amp(
        self,
        calc,
        system,
        amp_calc,
        test_traj,
        amp_test_traj,
        n_test=int(1e3),
        save_interval=10,
        size=(2, 2, 2),
        temp=300,
    ):
        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(test_traj, n_test, save_interval)
        generator.convert_traj(test_traj)

        amp_generator = GenerateTrajectory()
        generator.generate_system(amp_calc, system, size, temp)
        generator.create_traj(amp_test_traj, n_test, save_interval)
        generator.convert_traj(amp_test_traj)
