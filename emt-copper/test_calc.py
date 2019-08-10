import sys
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_error

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from analysis import Analyzer
from plotting import Plotter


if __name__ == "__main__":
    plter = Plotter()
    plter.plot_trainlog("calcs/energy-trained-log.txt", "energy_log.png")
    plter.plot_trainlog("calcs/force-trained-log.txt", "force_log.png")

    system = "copper"
    size = (2, 2, 2)
    temp = 300
    timestep = 5.0

    n_test = int(1e3)
    save_interval = 10

    trjbd = TrajectoryBuilder()
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)
    calc = Amp.load("calcs/force-trained.amp")
    amp_test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)

    test_traj = "test.traj"
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval, timestep=timestep, convert=True
    )

    amp_test_traj = "amp_test.traj"
    steps, amp_test_traj = trjbd.integrate_atoms(
        amp_test_atoms, amp_test_traj, n_test, save_interval, timestep=timestep, convert=True
    )

    legend = ["EMT", "AMP"]
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

    steps, msd = anl.calculate_msd(test_traj, save_interval=save_interval)
    steps, amp_msd = anl.calculate_msd(amp_test_traj, save_interval=save_interval)
    plter.plot_msd("msd.png", legend, steps, msd, amp_msd)

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
