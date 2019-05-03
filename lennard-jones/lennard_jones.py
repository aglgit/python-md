import os
import ase.io
from ase.lattice.cubic import FaceCenteredCubic
from ase import units
from ase.calculators.lj import LennardJones
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


def generate_data(
    n_steps, save_interval, symbol="Ar", size=(3, 3, 3), filename="training.traj"
):
    if os.path.exists(filename):
        return
    traj = ase.io.Trajectory(filename, "w")
    atoms = FaceCenteredCubic(symbol=symbol, size=size, pbc=True)
    MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
    atoms.set_calculator(LennardJones(sigma=3.405, epsilon=1.0318e-2))
    atoms.get_potential_energy()
    atoms.get_forces()
    traj.write(atoms)

    dyn = VelocityVerlet(atoms, 5 * units.fs)
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
