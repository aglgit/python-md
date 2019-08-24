import sys
from gpaw import GPAW
from amp import Amp
from amp.analysis import calculate_error

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from analysis import Analyzer
from plotting import Plotter


if __name__ == "__main__":
    system = "silicon"
    size = (2, 2, 2)
    temp = 300
    n_test = int(5e3)
    save_interval = 10
    timestep = 0.5
    legend = ["GPAW", "AMP"]

    energy_log = "energy-trained-log.txt"
    force_log = "force-trained-log.txt"
    energy_plot = system + "_" + "energy_log.png"
    force_plot = system + "_" + "force_log.png"
    plter = Plotter()
    plter.plot_trainlog(energy_log, energy_plot)
    plter.plot_trainlog(force_log, force_plot)

    trjbd = TrajectoryBuilder()
    calc = GPAW(mode="pw", symmetry={"point_group": False})
    test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)
    calc = Amp.load("calcs/force-trained.amp")
    amp_test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)

    test_traj = "test.traj"
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval, timestep=timestep, convert=True
    )

    amp_test_traj = "amp_test.traj"
    steps, amp_test_traj = trjbd.integrate_atoms(
        amp_test_atoms,
        amp_test_traj,
        n_test,
        save_interval,
        timestep=timestep,
        convert=True,
    )

    anl = Analyzer()
    r, rdf = anl.calculate_rdf(test_traj, r_max=6.0)
    r_amp, rdf_amp = anl.calculate_rdf(amp_test_traj, r_max=6.0)
    rdf_plot = system + "_" + "rdf.png"
    plter.plot_rdf(rdf_plot, legend, r, rdf, rdf_amp)

    steps, energy_exact, energy_amp = anl.calculate_pot_energy_diff(
        test_traj, amp_test_traj, save_interval=save_interval
    )
    pot_plot = system + "_" + "pot.png"
    plter.plot_pot_energy_diff(pot_plot, legend, steps, energy_exact, energy_amp)

    steps, energy_exact, energy_amp = anl.calculate_energy_diff(
        test_traj, amp_test_traj, save_interval=save_interval
    )
    energy_plot = system + "_" + "energy.png"
    plter.plot_energy_diff(energy_plot, legend, steps, energy_exact, energy_amp)

    steps, msd = anl.calculate_msd(test_traj, save_interval=save_interval)
    steps, amp_msd = anl.calculate_msd(amp_test_traj, save_interval=save_interval)
    msd_plot = system + "_" + "msd.png"
    plter.plot_msd(msd_plot, legend, steps, msd, amp_msd)

    energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
        "calcs/force-trained.amp", test_traj
    )
    plter.plot_amp_error(
        "energy_error.png",
        "force_error.png",
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )
