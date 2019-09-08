import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from amp.analysis import read_trainlog
from symmetry_functions import *


class Plotter:
    def __init__(self, plot_dir="plots", calc_dir="calcs"):
        sns.set()
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        self.plot_dir = plot_dir
        self.calc_dir = calc_dir

    def plot_trainlog(self, log_file, plot_file):
        log_file = os.path.join(self.calc_dir, log_file)
        plot_file = os.path.join(self.plot_dir, plot_file)

        log = read_trainlog(log_file)
        convergence = log["convergence"]

        steps = convergence["steps"]
        energy_rmse = convergence["es"]
        force_rmse = convergence["fs"]
        loss = convergence["costfxns"]

        plt.loglog(steps, energy_rmse, label="Energy RMSE")
        if force_rmse:
            plt.loglog(steps, force_rmse, label="Force RMSE")
            plt.ylabel("Error [eV, eV/Å]")
        else:
            plt.ylabel("Error [eV]")
        plt.loglog(steps, loss, label="Loss function")

        plt.title("Energy and force Root Mean Square Error")
        plt.legend()
        plt.xlabel("Steps")
        plt.savefig(plot_file)
        plt.clf()

    def plot_rdf(self, plot_file, legend, x, *rdfs):
        plot_file = os.path.join(self.plot_dir, plot_file)

        for rdf in rdfs:
            plt.plot(x, rdf)

        plt.title("Radial distribution function")
        plt.legend(legend)
        plt.xlabel("Radial distance [Å]")
        plt.ylabel("RDF")
        plt.savefig(plot_file)
        plt.clf()

    def plot_msd(self, plot_file, legend, steps, *msds):
        plot_file = os.path.join(self.plot_dir, plot_file)

        for msd in msds:
            plt.plot(steps, msd)

        plt.title("Mean Squared Displacement")
        plt.legend(legend)
        plt.xlabel("Steps")
        plt.ylabel("MSD [Å]")
        plt.savefig(plot_file)
        plt.clf()

    def plot_energy_diff(self, plot_file, legend, steps, energy_exact, energy_amp):
        plot_file = os.path.join(self.plot_dir, plot_file)

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
        plot_file = os.path.join(self.plot_dir, plot_file)

        plt.plot(steps, pot_energy_exact)
        plt.plot(steps, pot_energy_amp)

        plt.title("Potential energy as a function of time")
        plt.legend(legend)
        plt.xlabel("Steps")
        plt.ylabel("Potential energy [eV]")
        plt.savefig(plot_file)
        plt.clf()

    def plot_amp_error(
        self,
        energy_plot_file,
        force_plot_file,
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    ):
        energy_plot_file = os.path.join(self.plot_dir, energy_plot_file)
        force_plot_file = os.path.join(self.plot_dir, force_plot_file)

        plt.scatter(energy_exact, energy_diff)
        plt.axhline(y=energy_rmse, linestyle="--")
        plt.title("Scatterplot of energy error, energy RMSE={:.2E}".format(energy_rmse))
        plt.xlabel("Exact energy")
        plt.ylabel("abs(Exact energy - AMP energy) [eV]")
        plt.savefig(energy_plot_file)
        plt.clf()

        plt.scatter(force_exact, force_diff)
        plt.axhline(y=force_rmse, linestyle="--")
        plt.title("Scatterplot of force error, force RMSE={:.2E}".format(force_rmse))
        plt.xlabel("Exact force")
        plt.ylabel("abs(Exact force - AMP force) [eV/Å]")
        plt.savefig(force_plot_file)
        plt.clf()

    def plot_scaling(self, plot_file, num_atoms, times):
        plot_file = os.path.join(self.plot_dir, plot_file)

        plt.plot(num_atoms, times)
        plt.title("Time scaling as a function of system size")
        plt.xlabel("Number of atoms")
        plt.ylabel("Time [s]")
        plt.savefig(plot_file)
        plt.clf()

    def plot_symmetry_functions(
        self,
        rad_plot_file,
        ang_plot_file,
        Gs,
        cutoff=None,
        rij=None,
        rdf=None,
        theta=None,
        adf=None,
    ):
        rad_plot_file = os.path.join(self.plot_dir, rad_plot_file)
        ang_plot_file = os.path.join(self.plot_dir, ang_plot_file)

        if cutoff is None:
            cut = cosine
            r_cut = 6.0
        else:
            if cutoff.__class__.__name__ == "Cosine":
                cut = cosine
                r_cut = cutoff.Rc
            elif cutoff.__class__.__name__ == "Polynomial":
                cut = lambda rij, r_cut: polynomial(rij, r_cut, cutoff.gamma)
                r_cut = cutoff.Rc
            else:
                print("Cutoff {} not recognized".format(cutoff))
                raise NotImplementedError

        plt.figure(1)
        plt.plot(rij, cut(rij, r_cut), label="f_c", linestyle=":")
        if rij is None:
            rij = np.linspace(1e-3, r_cut, 1000)
        if rdf is not None:
            rdf[np.nonzero(rdf)] /= max(rdf)
            plt.plot(rij, rdf, label="rdf", linestyle="-.")
        plt.figure(2)
        if theta is None:
            theta = np.linspace(0, np.pi, 1000)
        if adf is not None:
            adf[np.nonzero(adf)] /= max(adf)
            plt.plot(theta, adf, label="adf", linestyle="--")

        for symm_func in Gs:
            if symm_func["type"] == "G2":
                eta = symm_func["eta"]
                center = symm_func["center"]
                val = G2(eta, center, rij, cut, r_cut)
                plt.figure(1)
                plt.plot(rij, val, label="eta={}, center={}".format(eta, center))
            elif symm_func["type"] in ["G4", "G5"]:
                eta = symm_func["eta"]
                gamma = symm_func["gamma"]
                zeta = symm_func["zeta"]
                val = G4(eta, gamma, zeta, theta)
                plt.figure(2)
                plt.plot(
                    theta,
                    val,
                    label="eta={}, gamma={}, zeta={}".format(eta, gamma, zeta),
                )

        plt.rc("legend", fontsize=5)
        plt.figure(1)
        plt.legend()
        plt.savefig(rad_plot_file, dpi=500)
        plt.clf()
        plt.figure(2)
        plt.legend()
        plt.savefig(ang_plot_file, dpi=500)
        plt.clf()
