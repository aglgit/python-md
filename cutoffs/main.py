import sys
from asap3 import EMT
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer


if __name__ == "__main__":
    system = "copper"
    elements = ["Cu"]
    size = (2, 2, 2)
    temp = 500
    n_train = int(8e4)
    n_test = int(2e4)
    save_interval = 100
    train_traj = "training.traj"
    test_traj = "test.traj"

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    num_radial_etas = 8
    num_angular_etas = 10
    num_zetas = 1
    angular_type = "G4"
    trn = Trainer(convergence=convergence, force_coefficient=force_coefficient)

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

    rcs = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    gamma = 5.0
    cutoffs = []
    for rc in rcs:
        cutoffs.append(Cosine(rc))
        cutoffs.append(Polynomial(rc, gamma=gamma))

    calcs = {}
    for cutoff in cutoffs:
        trn.cutoff = cutoff
        trn.create_Gs(
            elements, num_radial_etas, num_angular_etas, num_zetas, angular_type
        )

        label = "{}-{}".format(cutoff.__class__.__name__, cutoff.Rc)
        dblabel = label + "-train"
        calc = trn.create_calc(label=label, dblabel=dblabel)
        ann = Annealer(
            calc=calc,
            images=train_traj,
            Tmax=20,
            Tmin=1,
            steps=2000,
            train_forces=False,
        )
        amp_name = trn.train_calc(calc, train_traj)
        calcs[label] = amp_name

    columns = ["Cutoff", "Energy RMSE", "Force RMSE"]
    trn.test_calculators(calcs, test_traj, columns)
