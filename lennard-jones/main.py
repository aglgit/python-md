import seaborn as sns
from ase.calculators.lj import LennardJones
from analysis import Analyzer


if __name__ == "__main__":
    sns.set()

    calc = LennardJones(sigma=3.405, epsilon=1.0318e-2)
    system = "lennard_jones"
    train_filename = "training.traj"
    test_filename = "test.traj"
    amp_test_filename = "amp_test.traj"
    log_file = "amp-log.txt"
    fig_file = "convergence.png"

    anl = Analyzer()
    anl.train_amp(calc, system, train_filename)
    anl.plot_rmse(log_file, fig_file)
    anl.test_amp(calc, system, test_filename, amp_test_filename)

    rmax = 12.0
    nbins = 100
    x = (np.arange(nbins) + 0.5) * rmax / nbins
    rdf = anl.calculate_rdf(test_filename)
    amp_rdf = anl.calculate_rdf(amp_test_filename)
    plt.plot(x, rdf, label=system)
    plt.plot(x, amp_rdf, label="amp")
    plt.show()

    msd = anl.calculate_msd(test_filename)
    amp_msd = anl.calculate_msd(amp_test_filename)
    plt.plot(msd, label=system)
    plt.plot(msd, label="amp")
    plt.show()
