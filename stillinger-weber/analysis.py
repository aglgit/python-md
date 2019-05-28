import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ase.io
from amp import Amp
from amp.analysis import read_trainlog
from asap3 import OpenKIMcalculator
from asap3.analysis.rdf import RadialDistributionFunction
from generate_traj import GenerateTrajectory

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Analyzer:
    def __init__(self, log_file):
        self.log = read_trainlog(log_file)
        self.no_images = self.log["no_images"]
        self.convergence = self.log["convergence"]

    def plot_rmse(self, save_file):
        steps = self.convergence["steps"]
        energy_rmse = self.convergence["emrs"]
        force_rmse = self.convergence["fmrs"]
        loss = self.convergence["costfxns"]

        plt.semilogy(steps, energy_rmse, label="Energy RMSE")
        plt.semilogy(steps, force_rmse, label="Force RMSE")
        plt.semilogy(steps, loss, label="Loss function")

        plt.title("Energy and force Root Mean Square Error")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Error [eV, eV/Å]")
        plt.savefig(save_file)
        plt.clf()

    def generate_rdf(self, traj_file, rmax, nbins):
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


if __name__ == "__main__":
    train_filename = "training.traj"
    n_steps = 10000
    save_interval = 50
    size = (3, 3, 3)
    temp = 300
    calc = OpenKIMcalculator("SW_StillingerWeber_1985_Si__MO_405512056662_005")

    sw_train = GenerateTrajectory(calc)
    sw_train.stillinger_weber_system(size, temp)
    sw_train.create_traj(train_filename, n_steps, save_interval)

    print("Training from traj: {}".format(train_filename))
    traj = ase.io.read(train_filename, ":")
    descriptor = Gaussian(cutoff=6.0, fortran=True)
    convergence = {"energy_rmse": 1e-3, "force_rmse": 5e-2}
    loss_function = LossFunction(convergence=convergence)
    hidden_layers = (10, 10, 10)
    model = NeuralNetwork(activation="tanh", lossfunction=loss_function, hiddenlayers=hidden_layers)

    calc = Amp(descriptor=descriptor, model=model)
    calc.train(images=traj)

    sns.set()

    log_file = "amp-log.txt"
    save_file = "convergence.png"

    anl = Analyzer(log_file)
    anl.plot_rmse(save_file)

    sw_calc = OpenKIMcalculator("SW_StillingerWeber_1985_Si__MO_405512056662_005")
    sw_test_filename = "sw_test.traj"
    amp_calc = Amp.load("amp.amp")
    amp_test_filename = "amp_test.traj"

    sw_test = GenerateTrajectory(sw_calc)
    sw_test.stillinger_weber_system(size, temp)
    sw_test.create_traj(sw_test_filename, n_steps, save_interval)

    amp_test = GenerateTrajectory(amp_calc)
    amp_test.stillinger_weber_system(size, temp)
    amp_test.create_traj(amp_test_filename, n_steps, save_interval)

    rmax = 8.0
    nbins = 200
    x = (np.arange(nbins) + 0.5) * rmax / nbins
    lj_rdf = anl.generate_rdf(sw_test_filename, rmax, nbins)
    amp_rdf = anl.generate_rdf(amp_test_filename, rmax, nbins)

    plt.plot(x, lj_rdf, label="1")
    plt.plot(x, amp_rdf, label="2")
    plt.legend()
    plt.show()
