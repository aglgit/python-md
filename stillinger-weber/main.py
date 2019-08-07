import os
import sys
import numpy as np
from asap3 import OpenKIMcalculator
from amp import Amp
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial
from amp.descriptor.gaussian import make_symmetry_functions
from amp.model import LossFunction

sys.path.insert(1, "../tools")

from create_trajectory import TrajectoryBuilder
from analysis import Analyzer
from training import Trainer
from plotting import Plotter


if __name__ == "__main__":
    system = "silicon"
    size = (2, 2, 2)
    temp = 500

    n_train = int(8e4)
    n_train_force = int(5e3)
    save_interval = 100

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)

    elements = ["Si"]
    nr = 6
    nz = 1
    gammas = [1.0, -1.0]
    radial_etas = 10.0 * np.ones(nr)
    centers = np.linspace(0.5, cutoff.Rc + 0.5, nr)
    angular_etas = np.linspace(0.1, 1.0, nr)
    zetas = [4 ** i for i in range(nz)]
    G2 = make_symmetry_functions(
        elements=elements, type="G2", etas=radial_etas, centers=centers
    )
    G4 = make_symmetry_functions(
        elements=elements, type="G4", etas=angular_etas, zetas=zetas, gammas=[1.0, -1.0]
    )
    Gs = G2 + G4

    trn = Trainer(
        convergence=convergence,
        force_coefficient=force_coefficient,
        hidden_layers=hidden_layers,
        activation=activation,
        cutoff=cutoff,
        Gs=Gs,
    )

    trjbd = TrajectoryBuilder()
    calc = OpenKIMcalculator("SW_StillingerWeber_1985_Si__MO_405512056662_005")
    train_atoms = trjbd.build_atoms(system, size, temp, calc)

    train_traj = "training.traj"
    train_force_traj = "training_force.traj"
    steps, train_traj = trjbd.integrate_atoms(
        train_atoms, train_traj, n_train, save_interval, convert=True
    )
    steps, train_force_traj = trjbd.integrate_atoms(
        train_atoms,
        train_force_traj,
        n_train_force,
        save_interval,
        steps=steps,
        convert=True,
    )

    anl = Analyzer()
    plter = Plotter()
    r, rdf = anl.calculate_rdf(train_traj)
    plter.plot_symmetry_functions("rad.png", "ang.png", Gs, rij=r, rdf=rdf)

    label = "energy-trained"
    calc = trn.create_calc(label=label, dblabel=label)
    ann = Annealer(
        calc=calc, images=train_traj, Tmax=20, Tmin=1, steps=2000, train_forces=False
    )
    amp_name = trn.train_calc(calc, train_traj)

    label = os.path.join("calcs", "force-trained")
    calc = Amp.load(amp_name, label=label, dblabel=label)
    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": max_steps}
    loss_function = LossFunction(
        convergence=convergence, energy_coefficient=1.0, force_coefficient=0.1
    )
    calc.model.lossfunction = loss_function
    amp_name = trn.train_calc(calc, train_force_traj)
