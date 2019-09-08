from asap3 import EMT
from amp.descriptor.cutoffs import Cosine, Polynomial

from create_trajectory import TrajectoryBuilder
from training import Trainer

class ParameterSearch:
    def create_train_test(self):
        system = "copper"
        elements = ["Cu"]
        size = (2, 2, 2)
        temp = 500
        n_train = int(8e4)
        n_test = int(2e4)
        save_interval = 100
        train_traj = "training.traj"
        test_traj = "test.traj"

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

    def create_trainer(self):
        max_steps = int(2e3)
        convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
        force_coefficient = None
        cutoff = Polynomial(6.0, gamma=5.0)
        num_radial_etas = 7
        num_angular_etas = 10
        num_zetas = 1
        angular_type = "G4"
        trn = Trainer(
            convergence=convergence, force_coefficient=force_coefficient, cutoff=cutoff
        )
        trn.create_Gs(elements, num_radial_etas, num_angular_etas, num_zetas, angular_type)

        return trn

