import ase.io
import numpy as np
from asap3.analysis.rdf import RadialDistributionFunction
from generate_traj import GenerateTrajectory

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Analyzer:
    def __init__(self):
        self.n_train = int(1e5)
        self.n_test = int(1e4)
        self.save_interval = 25
        self.size = (3, 3, 3)
        self.temp = 1500

    def train_amp(self, calc, system, G, train_filename):
        xyz_filename = "".join((train_filename.split(".")[0], ".xyz"))
        convergence = {"energy_rmse": 1e-6}
        energy_coefficient = 1.0
        force_coefficient = None
        hidden_layers = (10, 10, 10)
        activation = "tanh"

        generator = GenerateTrajectory()
        generator.generate_system(calc, system, self.size, self.temp)
        generator.create_traj(train_filename, self.n_train, self.save_interval)
        generator.convert_traj(train_filename, xyz_filename)

        print("Training from traj: {}".format(train_filename))

        traj = ase.io.read(train_filename, ":")
        descriptor = Gaussian(Gs=G, fortran=True)
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

    def calculate_rdf(self, traj_file):
        rmax = 10.0
        nbins = 100
        x = (np.arange(nbins) + 0.5) * rmax / nbins

        traj = ase.io.read(traj_file, ":")
        rdf_obj = None
        for atoms in traj:
            if rdf_obj is None:
                rdf_obj = RadialDistributionFunction(atoms, rmax, nbins)
            else:
                rdf_obj.atoms = atoms
            rdf_obj.update()
        rdf = rdf_obj.get_rdf()

        return x, rdf

    def calculate_msd(self, traj_file):
        traj = ase.io.read(traj_file, ":")
        steps = np.arange(len(traj))
        msd = np.zeros(len(traj))
        init_pos = traj[0].get_positions()
        for i, atoms in enumerate(traj[1:]):
            pos = atoms.get_positions()
            disp = pos - init_pos
            msd[i + 1] = msd[i - 1] + np.linalg.norm(disp, axis=1).sum()

        return steps, msd

    def calculate_energy_diff(self, test_traj_file, amp_traj_file):
        test_traj = ase.io.read(test_traj_file, ":")
        amp_traj = ase.io.read(amp_traj_file, ":")

        num_images = len(test_traj)
        steps = np.arange(num_images)
        energy_exact = np.zeros(num_images)
        energy_amp = np.zeros(num_images)
        for i in range(num_images):
            energy_exact[i] = test_traj[i].get_potential_energy()
            energy_amp[i] = amp_traj[i].get_potential_energy()

        return steps, energy_exact, energy_amp

    def calculate_force_diff(self, test_traj_file, amp_traj_file):
        test_traj = ase.io.read(test_traj_file, ":")
        amp_traj = ase.io.read(amp_traj_file, ":")

        num_images = len(test_traj)
        num_forces = len(test_traj[0].get_forces())
        force_exact = np.zeros((num_images, num_forces))
        force_amp = np.zeros((num_images, num_forces))
        for i in range(num_images):
            force_exact[i] = test_traj[i].get_forces()
            force_amp[i] = amp_traj[i].get_forces()

        return force_exact, force_amp
