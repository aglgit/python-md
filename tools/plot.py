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
        plt.semilogy(steps, force_rmse, label="Force RMSE")
        plt.semilogy(steps, loss, label="Loss function")

        plt.title("Energy and force Root Mean Square Error")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Error [eV, eV/Å]")
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
