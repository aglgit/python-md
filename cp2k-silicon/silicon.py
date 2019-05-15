import os
import ase.io
from ase.lattice.cubic import Diamond
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.calculators.cp2k import CP2K

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


def generate_data(
    n_steps, save_interval, symbol="Si", size=(2, 2, 2), filename="training.traj"
):
    if os.path.exists(filename):
        return
    traj = ase.io.Trajectory(filename, "w")
    atoms = Diamond(symbol=symbol, size=size, pbc=True)
    MaxwellBoltzmannDistribution(atoms, 300 * units.kB)

    calc = CP2K()
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    atoms.get_kinetic_energy()
    atoms.get_forces()
    traj.write(atoms)

    dyn = VelocityVerlet(atoms, dt=units.fs)
    count = n_steps // save_interval
    print("Generating traj: {}".format(filename))
    print("Timestep: 0")
    for i in range(count):
        dyn.run(save_interval)
        atoms.get_potential_energy()
        atoms.get_kinetic_energy()
        atoms.get_forces()
        traj.write(atoms)
        print("Timestep: {}".format((i + 1) * save_interval))


filename = "training.traj"
generate_data(1000, 10, filename=filename)

print("Training from traj: {}".format(filename))
traj = ase.io.read(filename, ":")
calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
calc.model.lossfunction = LossFunction(
    convergence={"energy_rmse": 1E-3, "force_rmse": 1E-2}
)
calc.train(images=traj)
