import sys

sys.path.insert(0, "../tools")

import os
from asap3 import OpenKIMcalculator
from analysis import Analyzer
from plot import Plotter


if __name__ == "__main__":
    calc = OpenKIMcalculator("SW_StillingerWeber_1985_Si__MO_405512056662_005")
    system = "stillinger_weber"
    train_filename = "training.traj"
    test_filename = "test.traj"
    amp_test_filename = "amp_test.traj"
    log_file = "amp-log.txt"
    conv_file = "convergence_{}.png".format(system)
    rdf_file = "rdf_{}.png".format(system)
    msd_file = "msd_{}.png".format(system)
    energy_file = "energy_{}.png".format(system)
    Gs = None

    anl = Analyzer()
    plt = Plotter()
    if not os.path.exists("amp.amp"):
        anl.train_amp(calc, system, Gs, train_filename)
    plt.plot_rmse(log_file, conv_file)
    anl.test_amp(calc, system, test_filename, amp_test_filename)

    x, rdf = anl.calculate_rdf(test_filename)
    x, amp_rdf = anl.calculate_rdf(amp_test_filename)
    legend = ["Stillinger-Weber", "AMP"]
    plt.plot_rdf(rdf_file, legend, x, rdf, amp_rdf)

    steps, msd = anl.calculate_msd(test_filename)
    steps, amp_msd = anl.calculate_msd(amp_test_filename)
    plt.plot_msd(msd_file, legend, steps, msd, amp_msd)

    steps, energy_exact, energy_amp = anl.calculate_energy_diff(
        test_filename, amp_test_filename
    )
    plt.plot_energy_diff(energy_file, legend, steps, energy_exact, energy_amp)
