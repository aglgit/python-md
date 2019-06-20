import sys

sys.path.insert(0, "../tools")

import os
from ase.calculators.lj import LennardJones
from analysis import Analyzer
from plot import Plotter


if __name__ == "__main__":
    calc = LennardJones(sigma=3.405, epsilon=1.0318e-2)
    system = "lennard_jones"
    train_filename = "training.traj"
    test_filename = "test.traj"
    amp_test_filename = "amp_test.traj"
    log_file = "amp-log.txt"
    conv_file = "convergence.png"
    rdf_file = "rdf.png"
    msd_file = "msd.png"
    energy_file = "energy.png"
    G = None

    anl = Analyzer()
    plt = Plotter()
    if not os.path.exists("amp.amp"):
        anl.train_amp(calc, system, G, train_filename)
    plt.plot_rmse(log_file, conv_file)
    anl.test_amp(calc, system, test_filename, amp_test_filename)

    x, rdf = anl.calculate_rdf(test_filename)
    x, amp_rdf = anl.calculate_rdf(amp_test_filename)
    legend = ["Lennard-Jones", "AMP"]
    plt.plot_rdf(rdf_file, legend, x, rdf, amp_rdf)

    steps, msd = anl.calculate_msd(test_filename)
    steps, amp_msd = anl.calculate_msd(amp_test_filename)
    plt.plot_msd(msd_file, legend, steps, msd, amp_msd)

    steps, energy_exact, energy_amp = anl.calculate_energy_diff(
        test_filename, amp_test_filename
    )
    plt.plot_energy_diff(energy_file, steps, energy_exact, energy_amp)
