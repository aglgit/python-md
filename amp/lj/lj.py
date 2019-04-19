import amp
import ase
import os
import ase.io
from ase import Atoms, Atom, units
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.lj import LennardJones
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork

def generate_data(count, filename="lj.traj"):
    if os.path.exists(filename):
        return
    traj = ase.io.Trajectory(filename, "w")
    size = 5
    b = 3.0
    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol="Ar",
                              size=(size, size, size),
                              latticeconstant=b,
                              pbc=True)
    calc = LennardJones()
    atoms.set_calculator(calc)
    MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
    dyn = VelocityVerlet(atoms, units.fs)

    print("Creating training data {}".format(filename))
    print("Timestep: {} Energy: {:f}".format(0, atoms.get_total_energy()))
    for i in range(count - 1):
        dyn.run(10)
        traj.write(atoms)
        print("Timestep: {} Energy: {:f}".format((i+1)*10, atoms.get_total_energy()))

generate_data(10)

calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
calc.train(images="lj.traj")
