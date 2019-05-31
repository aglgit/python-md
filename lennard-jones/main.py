import seaborn as sns
from ase.calculators.lj import LennardJones
from analysis import Analyzer


if __name__ == "__main__":
    sns.set()

    calc = LennardJones(sigma=3.405, epsilon=1.0318e-2)
    system = "lennard_jones"
    train_filename = "training.traj"
    test_filename = "test.traj"
    log_file = "amp-log.txt"
    fig_file = "convergence.png"

    anl = Analyzer()
    anl.train_amp(calc, system, train_filename)
    anl.plot_rmse(log_file, fig_file)
