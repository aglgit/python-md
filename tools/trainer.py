import ase.io
from generate_traj import GenerateTrajectory

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Trainer:
    def __init__(
        self,
        n_train=int(1e4),
        n_test=int(1e3),
        save_interval=50,
        size=(3, 3, 3),
        temp=1500,
    ):
        self.n_train = n_train
        self.n_test = n_test
        self.save_interval = save_interval
        self.size = size
        self.temp = temp

    def train_amp(
        self,
        calc,
        system,
        train_filename="training.traj",
        convergence=None,
        energy_coefficient=1.0,
        force_coefficient=None,
        hidden_layers=(10, 10, 10),
        activation="tanh",
        Gs=None,
    ):
        xyz_filename = "".join((train_filename.split(".")[0], ".xyz"))
        if convergence is None:
            convergence = {"energy_rmse": 1e-6}

        generator = GenerateTrajectory()
        generator.generate_system(calc, system, self.size, self.temp)
        generator.create_traj(train_filename, self.n_train, self.save_interval)
        generator.convert_traj(train_filename, xyz_filename)

        print("Training from traj: {}".format(train_filename))

        traj = ase.io.read(train_filename, ":")
        descriptor = Gaussian(Gs=Gs, fortran=True)
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

        calc = Amp(descriptor=descriptor, model=model)
        calc.train(images=traj)

    def test_amp(self, calc, system, test_filename, amp_test_filename):
        amp_calc = Amp.load("amp.amp")

        xyz_test_filename = "".join((test_filename.split(".")[0], ".xyz"))
        generator = GenerateTrajectory()
        generator.generate_system(calc, system, self.size, self.temp)
        generator.create_traj(test_filename, self.n_test, self.save_interval)
        generator.convert_traj(test_filename, xyz_test_filename)

        xyz_amp_test_filename = "".join((amp_test_filename.split(".")[0], ".xyz"))
        amp_generator = GenerateTrajectory()
        amp_generator.generate_system(amp_calc, system, self.size, self.temp)
        amp_generator.create_traj(amp_test_filename, self.n_test, self.save_interval)
        amp_generator.convert_traj(amp_test_filename, xyz_amp_test_filename)
