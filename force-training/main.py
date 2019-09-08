import sys
from asap3 import EMT
from amp.utilities import Annealer
from amp.analysis import calculate_error
from amp.descriptor.cutoffs import Cosine, Polynomial

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    system = "copper"
    elements = ["Cu"]
    size = (3, 3, 3)
    temp = 500
    n_train = int(5e4)
    n_test = int(1e4)
    save_interval = 100
    train_traj = "training.traj"
    test_traj = "test.traj"

    max_steps = int(2e3)
    cutoff = Polynomial(6.0, gamma=5.0)
    num_radial_etas = 8
    num_angular_etas = 10
    num_zetas = 1
    angular_type = "G4"
    trn = Trainer(cutoff=cutoff)
    trn.create_Gs(elements, num_radial_etas, num_angular_etas, num_zetas, angular_type)

    trjbd = TrajectoryBuilder()
    calc = EMT()
    train_atoms = trjbd.build_atoms(system, size, temp, calc)
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc)

    steps, train_traj = trjbd.integrate_atoms(
        train_atoms, train_traj, n_train, save_interval
    )
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval
    )

    plter = Plotter()
    energy_noforcetrain = "energy_noforcetrain.png"
    force_noforcetrain = "force_noforcetrain.png"
    energy_forcetrain = "energy_forcetrain.png"
    force_forcetrain = "force_forcetrain.png"

    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    trn.convergence = convergence
    trn.force_coefficient = force_coefficient
    label = "energy"
    dblabel = label + "-train"
    calc = trn.create_calc(label=label, dblabel=dblabel)
    ann = Annealer(
        calc=calc, images=train_traj, Tmax=20, Tmin=1, steps=2000, train_forces=False
    )
    energy_amp_name = trn.train_calc(calc, train_traj)

    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    force_coefficient = 0.1
    trn.convergence = convergence
    trn.force_coefficient = force_coefficient
    label = "force"
    dblabel = label + "-train"
    calc = trn.create_calc(label=label, dblabel=dblabel)
    ann = Annealer(
        calc=calc, images=train_traj, Tmax=20, Tmin=1, steps=2000, train_forces=True
    )
    force_amp_name = trn.train_calc(calc, train_traj)

    label = calc.label
    dblabel = label + "-test"
    energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
        energy_amp_name, images=test_traj, label=label, dblabel=dblabel
    )
    plter.plot_amp_error(
        energy_noforcetrain,
        force_noforcetrain,
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )

    label = calc.label
    dblabel = label + "-test"
    energy_rmse, force_rmse, energy_exact, energy_diff, force_exact, force_diff = calculate_error(
        force_amp_name, images=test_traj, label=label, dblabel=dblabel
    )
    plter.plot_amp_error(
        energy_forcetrain,
        force_forcetrain,
        energy_rmse,
        force_rmse,
        energy_exact,
        energy_diff,
        force_exact,
        force_diff,
    )
