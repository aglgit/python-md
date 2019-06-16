import sys

sys.path.insert(0, "../tools")

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
    conv_file = "convergence.png"
    rdf_file = "rdf.png"

    anl = Analyzer()
    plt = Plotter()
    anl.train_amp(calc, system, train_filename)
    plt.plot_rmse(log_file, conv_file)
    anl.test_amp(calc, system, test_filename, amp_test_filename)

    x, rdf = anl.calculate_rdf(test_filename)
    x, amp_rdf = anl.calculate_rdf(amp_test_filename)
    legend = ["Stillinger-Weber", "AMP"]
    plt.plot_rdf(rdf_file, legend, x, rdf, amp_rdf)
