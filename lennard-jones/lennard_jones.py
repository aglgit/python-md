import os
import numpy as np

from ase import units
from ase.io import Trajectory
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import SimpleCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


def generate_data(count, timesteps, filename="training.traj"):
    if os.path.exists(filename):
        return
    traj = Trajectory(filename, "w")
    size = 10
    lattice_constant = 5.260
    argon = SimpleCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=(size, size, size),
                          symbol='Ar', pbc=True, latticeconstant=lattice_constant)
    argon.set_calculator(LennardJones(sigma=3.405, epsilon=1.0318e-2))
    argon.get_potential_energy()
    MaxwellBoltzmannDistribution(argon, 300 * units.kB)
    traj.write(argon)
    dyn = VelocityVerlet(argon, 5 * units.fs)  # 1 fs time step.
    print("Starting MD simulation: {}".format(filename))
    print("Timestep: 0")
    for i in range(count):
        dyn.run(timesteps)
        print("Timestep: {}".format((i+1)*timesteps))
        traj.write(argon)

count = 200
timesteps = 50
filename = "training.traj"
generate_data(count, timesteps, filename)

print("Training images {}".format(filename))
calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
calc.model.lossfunction = LossFunction(convergence={"energy_rmse": 0.02,
                                                    "force_rmse": 0.02})
calc.train(images='training.traj')
