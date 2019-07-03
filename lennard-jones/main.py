import sys

sys.path.insert(0, "../tools")

import os
from ase.calculators.lj import LennardJones
from analysis import Analyzer
from plot import Plotter
from trainer import Trainer

if __name__ == "__main__":
    sigma = 3.405
    epsilon = 1.0318e-2
    calc = LennardJones(sigma=sigma, epsilon=epsilon)
    system = "lennard_jones"

    train_filename = "training.traj"
    test_filename = "test.traj"
    amp_test_filename = "amp_test.traj"
    log_file = "amp-log.txt"
    conv_file = "convergence_{}.png".format(system)
    rdf_file = "rdf_{}.png".format(system)
    msd_file = "msd_{}.png".format(system)
    energy_file = "energy_{}.png".format(system)

    n_train = int(1e5)
    size = (5, 5, 5)
    save_interval = 10
    temp = 300

    convergence = {"energy_rmse": 1e-6, "max_steps": int(1e4)}
    cutoff = 4.0
    Gs = None

    trn = Trainer(n_train=n_train, size=size, save_interval=save_interval, temp=temp)
    if not os.path.exists("amp.amp"):
        trn.train_amp(calc, system, convergence=convergence, cutoff=cutoff, Gs=Gs)

    anl = Analyzer(save_interval=save_interval)
    plt = Plotter()
    plt.plot_rmse(log_file, conv_file)
    trn.test_amp(calc, system, test_filename, amp_test_filename)

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
    plt.plot_energy_diff(energy_file, legend, steps, energy_exact, energy_amp)
