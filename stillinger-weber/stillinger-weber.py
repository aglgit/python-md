import os
import quippy
import numpy as np

from ase.io import Trajectory

from quippy import Potential, supercell, diamond, DynamicalSystem

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


def generate_data(n_steps, timestep, filename="training.traj"):
    if os.path.exists(filename):
        return
    traj = Trajectory(filename, "w")
    size = 10
    lattice_constant = 5.44
    atomic_number = 14
    silicon = supercell(diamond(lattice_constant, atomic_number), size, size, size)
    pot = Potential("IP SW")
    silicon.set_calculator(pot)
    silicon.set_cutoff(pot.cutoff() + 2.0)

    ds = DynamicalSystem(silicon)
    ds.rescale_velo(300.0)
    connect_interval = 10

    ds.atoms.calc_connect()
    pot.calc(ds.atoms, energy=True, force=True)
    ds.print_status(epot=ds.atoms.energy)
    traj.write(ds.atoms, calculator=pot)

    for i in range(1, n_steps+1):
        ds.advance_verlet1(timestep)
        pot.calc(ds.atoms, energy=True, force=True)
        ds.advance_verlet2(timestep, ds.atoms.force)
        if i % connect_interval == 0:
            ds.print_status(epot=ds.atoms.energy)
            ds.atoms.calc_connect()
            traj.write(ds.atoms)

n_steps = 100
timestep = 1.0
filename = "training.traj"
generate_data(n_steps, timestep, filename)

print("Training images {}".format(filename))
calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
calc.model.lossfunction = LossFunction(convergence={"energy_rmse": 0.01,
                                                    "force_rmse": 0.01})
calc.train(images='training.traj')
