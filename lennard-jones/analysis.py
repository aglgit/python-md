import numpy as np
import matplotlib.pyplot as plt
import ase.io
from asap3.analysis.rdf import RadialDistributionFunction
from generate_traj import GenerateTrajectory

from amp import Amp
from amp.analysis import read_trainlog
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Analyzer:
    def train_amp(self, calc, system, train_filename):
        xyz_filename = "".join((train_filename.split(".")[0], ".xyz"))
        n_steps = 10000
        save_interval = 50
        size = (3, 3, 3)
        temp = 300

        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(train_filename, n_steps, save_interval)
        generator.convert_traj(train_filename, xyz_filename)

        print("Training from traj: {}".format(train_filename))

        traj = ase.io.read(train_filename, ":")
        convergence = {"energy_rmse": 1e-5, "force_rmse": 5e-3}
        energy_coefficient = 1.0
        force_coefficient = 0.04
        hidden_layers = (10, 10, 10)
        activation = "tanh"

        descriptor = Gaussian(fortran=True)
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
        n_steps = 10000
        save_interval = 50
        size = (3, 3, 3)
        temp = 300

        generator = GenerateTrajectory()
        generator.generate_system(calc, system, size, temp)
        generator.create_traj(train_filename, n_steps, save_interval)

        amp_generator = GenerateTrajectory()
        amp_generator.generate_system(amp_calc, system, size, temp)
        amp_generator.create_traj(amp_test_filename, n_steps, save_interval)

    def plot_rmse(self, log_file, plot_file):
        log = read_trainlog(log_file)
        convergence = log["convergence"]

        steps = convergence["steps"]
        energy_rmse = convergence["es"]
        force_rmse = convergence["fs"]
        loss = convergence["costfxns"]

        plt.semilogy(steps, energy_rmse, label="Energy RMSE")
        plt.semilogy(steps, force_rmse, label="Force RMSE")
        plt.semilogy(steps, loss, label="Loss function")

        plt.title("Energy and force Root Mean Square Error")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Error [eV, eV/Å]")
        plt.savefig(plot_file)
        plt.clf()

    def calculate_rdf(self, traj_file, rmax, nbins):
        traj = ase.io.read(traj_file, ":")
        rdf_obj = None
        for atoms in traj:
            if rdf_obj is None:
                rdf_obj = RadialDistributionFunction(atoms, rmax, nbins)
            else:
                rdf_obj.atoms = atoms
            rdf_obj.update()
        rdf = rdf_obj.get_rdf()

        return rdf

    def calculate_msd(self, traj_file):
        traj = ase.io.read(traj_file, ":")
        init_pos = traj[0].get_positions()
        cell = traj[0].get_cell()
        msd = np.zeros(len(traj))
        for atoms in traj:
            pos = atoms.get_positions()
            disp = pos - init_pos
            msd += np.linalg.norm(disp, axis=1).sum()

        return msd

    def calculate_energy_diff(self, test_traj_file, amp_traj_file):
        test_traj = ase.io.read(test_traj_file, ":")
        amp_traj = ase.io.read(amp_traj_file, ":")

        num_images = len(test_traj)
        energy_exact = np.zeros(num_images)
        energy_amp = np.zeros(num_images)
        for i in range(num_images):
            energy_exact[i] = test_traj[i].get_potential_energy()
            energy_amp[i] = amp_traj[i].get_potential_energy()

        return energy_exact, energy_amp

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
