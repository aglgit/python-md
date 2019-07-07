import matplotlib.pyplot as plt
import seaborn as sns
from amp.analysis import read_trainlog


class Plotter:
    def __init__(self):
        sns.set()

    def plot_rmse(self, log_file, plot_file):
        log = read_trainlog(log_file)
        convergence = log["convergence"]

        steps = convergence["steps"]
        energy_rmse = convergence["es"]
        force_rmse = convergence["fs"]
        loss = convergence["costfxns"]

        plt.semilogy(steps, energy_rmse, label="Energy RMSE")
        if force_rmse:
            plt.semilogy(steps, force_rmse, label="Force RMSE")
            plt.ylabel("Error [eV, eV/Å]")
        else:
            plt.ylabel("Error [eV]")
        plt.semilogy(steps, loss, label="Loss function")

        plt.title("Energy and force Root Mean Square Error")
        plt.legend()
        plt.xlabel("Steps")
        plt.savefig(plot_file)
        plt.clf()

    def plot_rdf(self, plot_file, legend, x, *rdfs):
        for rdf in rdfs:
            plt.plot(x, rdf)

        plt.title("Radial distribution function")
        plt.legend(legend)
        plt.xlabel("Radial distance [Å]")
        plt.ylabel("RDF")
        plt.savefig(plot_file)
        plt.clf()

    def plot_msd(self, plot_file, legend, steps, *msds):
        for msd in msds:
            plt.plot(steps, msd)

        plt.title("Mean Squared Displacement")
        plt.legend(legend)
        plt.xlabel("Steps")
        plt.ylabel("MSD [Å]")
        plt.savefig(plot_file)
        plt.clf()

    def plot_energy_diff(self, plot_file, legend, steps, energy_exact, energy_amp):
        plt.plot(steps, energy_exact)
        plt.plot(steps, energy_amp)

        plt.title("Total energy as a function of time")
        plt.legend(legend)
        plt.xlabel("Steps")
        plt.ylabel("Total energy [eV]")
        plt.savefig(plot_file)
        plt.clf()

    def plot_pot_energy_diff(
        self, plot_file, legend, steps, pot_energy_exact, pot_energy_amp
    ):
        plt.plot(steps, pot_energy_exact)
        plt.plot(steps, pot_energy_amp)

        plt.title("Potential energy as a function of time")
        plt.legend(legend)
        plt.xlabel("Steps")
        plt.ylabel("Potential energy [eV]")
        plt.savefig(plot_file)
        plt.clf()
