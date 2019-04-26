import os
import ase.io
from ase.lattice.cubic import FaceCenteredCubic
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from asap3 import EMT

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


def generate_data(
    n_steps, save_interval, symbol="Cu", size=(3, 3, 3), filename="training.traj"
):
    if os.path.exists(filename):
        return
    traj = ase.io.Trajectory(filename, "w")
    atoms = FaceCenteredCubic(symbol=symbol, size=size, pbc=True)
    MaxwellBoltzmannDistribution(atoms, 300 * units.kB)

    atoms.set_calculator(EMT())
    atoms.get_potential_energy()
    atoms.get_forces()
    traj.write(atoms)

    dyn = VelocityVerlet(atoms, dt=units.fs)
    count = n_steps // save_interval
    print("Generating traj: {}".format(filename))
    print("Timestep: 0")
    for i in range(count):
        dyn.run(save_interval)
        atoms.get_potential_energy()
        atoms.get_forces()
        traj.write(atoms)
        print("Timestep: {}".format((i + 1) * save_interval))


filename = "training.traj"
generate_data(1000, 10, filename=filename)

print("Training from traj: {}".format(filename))
traj = ase.io.read(filename, ":")
calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
calc.model.lossfunction = LossFunction(
    convergence={"energy_rmse": 0.02, "force_rmse": 0.02}
)
calc.train(images=traj)
