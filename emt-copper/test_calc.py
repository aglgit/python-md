import os
import sys
import numpy as np
from asap3 import EMT
from amp import Amp
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import make_symmetry_functions
from amp.model import LossFunction

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from analysis import Analyzer
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    plter = Plotter()
    plter.plot_trainlog("calcs/energy-trained-log.txt", "energy_log.png")
    plter.plot_trainlog("calcs/force-trained-log.txt", "force_log.png")

    system = "copper"
    size = (2, 2, 2)
    temp = 300

    n_test = int(1e3)
    save_interval = 10

    trjbd = TrajectoryBuilder()
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)
    calc = Amp.load("calcs/force-trained.amp")
    amp_test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)

    test_traj = "test.traj"
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval, convert=True
    )

    amp_test_traj = "amp_test.traj"
    steps, amp_test_traj = trjbd.integrate_atoms(
        amp_test_atoms, amp_test_traj, n_test, save_interval, convert=True
    )

    legend = ["Test", "AMP"]
    anl = Analyzer()
    r, rdf = anl.calculate_rdf(test_traj, r_max=6.0)
    r_amp, rdf_amp = anl.calculate_rdf(amp_test_traj, r_max=6.0)
    plter.plot_rdf("rdf.png", legend, r, rdf, rdf_amp)

    steps, energy_exact, energy_amp = anl.calculate_pot_energy_diff(
        test_traj, amp_test_traj, save_interval=save_interval
    )
    plter.plot_pot_energy_diff(
        "pot_energy.png", legend, steps, energy_exact, energy_amp
    )

    steps, energy_exact, energy_amp = anl.calculate_energy_diff(
        test_traj, amp_test_traj, save_interval=save_interval
    )
    plter.plot_energy_diff("energy.png", legend, steps, energy_exact, energy_amp)
