import sys
from asap3 import EMT
from amp.utilities import Annealer

sys.path.insert(1, "../tools")

from parameter_search import ParameterSearch
from create_trajectory import TrajectoryBuilder
from training import Trainer


if __name__ == "__main__":
    pms = ParameterSearch()
    trn = pms.create_trainer()

    system = "copper"
    elements = ["Cu"]
    size = (2, 2, 2)
    temp = 500
    n_test = int(2e4)
    save_interval = 100

    trjbd = TrajectoryBuilder()
    n_images = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    train_trajs = ["training_n{}.traj".format(ni) for ni in n_images]
    for i in range(len(n_images)):
        calc = EMT()
        train_atoms = trjbd.build_atoms(system, size, temp, calc)
        n_train = n_images[i] * save_interval
        steps, train_trajs[i] = trjbd.integrate_atoms(
            train_atoms, train_trajs[i], n_train, save_interval
        )

    test_traj = "test.traj"
    calc = EMT()
    test_atoms = trjbd.build_atoms(system, size, temp, calc)
    steps, test_traj = trjbd.integrate_atoms(
        test_atoms, test_traj, n_test, save_interval
    )

    calcs = {}
    for i in range(len(n_images)):
        label = "n{}".format(n_images[i])
        dblabel = label + "-train"
        calc = trn.create_calc(label=label, dblabel=dblabel)
        ann = Annealer(
            calc=calc,
            images=train_trajs[i],
            Tmax=20,
            Tmin=1,
            steps=2000,
            train_forces=False,
        )
        amp_name = trn.train_calc(calc, train_trajs[i])
        calcs[label] = amp_name

    columns = ["Number of images", "Energy RMSE", "Force RMSE"]
    trn.test_calculators(calcs, test_traj, columns)
