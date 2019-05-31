import os
from ase import Atoms, Atom, units
import ase.io
from ase.lattice.surface import fcc110
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from asap3 import EMT

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


def generate_data(count, filename="training.traj"):
    """Generates test or training data with a simple MD simulation."""
    if os.path.exists(filename):
        return
    traj = ase.io.Trajectory(filename, "w")
    atoms = fcc110("Pt", (2, 2, 2), vacuum=7.0)
    atoms.extend(
        Atoms(
            [
                Atom("Cu", atoms[7].position + (0.0, 0.0, 2.5)),
                Atom("Cu", atoms[7].position + (0.0, 0.0, 5.0)),
            ]
        )
    )
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    atoms.set_calculator(EMT())
    atoms.get_potential_energy()
    atoms.get_forces()
    traj.write(atoms)
    MaxwellBoltzmannDistribution(atoms, 300.0 * units.kB)
    dyn = VelocityVerlet(atoms, dt=1.0 * units.fs)
    for step in range(count - 1):
        dyn.run(50)
        atoms.get_potential_energy()
        atoms.get_forces()
        traj.write(atoms)


generate_data(20)

calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
calc.model.lossfunction = LossFunction(
    convergence={"energy_rmse": 0.02, "force_rmse": 0.02}
)
calc.train(images="training.traj")