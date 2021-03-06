import sys
import time
import numpy as np
from asap3 import EMT
from amp import Amp
from amp.analysis import calculate_error

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from analysis import Analyzer
from plotting import Plotter


if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 300
    timestep = 1.0
    n_test = int(5e3)
    save_interval = 10
    legend = ["EMT", "AMP"]

    energy_log = "energy-trained-log.txt"
    force_log = "force-trained-log.txt"
    energy_plot = system + "_" + "energy_log.png"
    force_plot = system + "_" + "force_log.png"
    plter = Plotter()
    plter.plot_trainlog(energy_log, energy_plot)
    plter.plot_trainlog(force_log, force_plot)

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
        "calcs/force-trained.amp", test_traj, label="amp", dblabel="amp"
    )
    plter.plot_amp_error(
        system + "_" + "energy_error.png",
        system + "_" + "force_error.png",
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )

    test_sizes = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (2, 2, 3),
        (2, 3, 3),
        (3, 3, 3),
        (3, 3, 4),
        (3, 4, 4),
        (4, 4, 4),
    ]
    num_atoms = np.zeros(len(test_sizes))
    times = np.zeros(len(test_sizes))
    for i, size in enumerate(test_sizes):
        n = 4
        for dim in size:
            n *= dim
        num_atoms[i] = n
        calc = Amp.load("calcs/force-trained.amp")
        test_atoms = trjbd.build_atoms(system, size, temp, calc, seed=0)
        start = time.time()
        test_atoms.get_forces()
        end = time.time()
        times[i] = end - start

    plter.plot_scaling(system + "_" + "scaling.png", num_atoms, times)
