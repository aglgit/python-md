import sys
from asap3 import EMT
from amp import Amp
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.model import LossFunction

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    system = "copper"
    elements = ["Cu"]
    size = (2, 2, 2)
    temp = 500
    n_train = int(8e5)
    n_train_force = int(5e4)
    save_interval = 100

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    cutoff = Polynomial(6.0, gamma=5.0)
    num_radial_etas = 6
    num_angular_etas = 10
    num_zetas = 1
    angular_type = "G4"
    trn = Trainer(
        convergence=convergence, force_coefficient=force_coefficient, cutoff=cutoff
    )
    trn.create_Gs(elements, num_radial_etas, num_angular_etas, num_zetas, angular_type)

    trjbd = TrajectoryBuilder()
    calc = EMT()
    train_atoms = trjbd.build_atoms(system, size, temp, calc)

    train_traj = "training.traj"
    train_force_traj = "training_force.traj"
    steps, train_traj = trjbd.integrate_atoms(
        train_atoms, train_traj, n_train, save_interval
    )
    steps, train_force_traj = trjbd.integrate_atoms(
        train_atoms, train_force_traj, n_train_force, save_interval, steps=steps
    )

    label = "energy-trained"
    dblabel = label + "-train"
    calc = trn.create_calc(label=label, dblabel=dblabel)
    ann = Annealer(
        calc=calc, images=train_traj, Tmax=20, Tmin=1, steps=2000, train_forces=False
    )
    amp_name = trn.train_calc(calc, train_traj)

    label = os.path.join("calcs", "force-trained")
    dblabel = label + "-train"
    calc = Amp.load(amp_name, label=label, dblabel=dblabel)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    loss_function = LossFunction(
        convergence=convergence, energy_coefficient=1.0, force_coefficient=0.05
    )
    calc.model.lossfunction = loss_function
    amp_name = trn.train_calc(calc, train_force_traj)
