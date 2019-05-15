import os
import ase.io
from ase.lattice.cubic import FaceCenteredCubic
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from amp import Amp


def test_calc(
    n_steps, save_interval, symbol="Ar", size=(4, 4, 4), filename="test.traj"
):
    if os.path.exists(filename):
        return
    traj = ase.io.Trajectory(filename, "w")
    atoms = FaceCenteredCubic(symbol=symbol, size=size, pbc=True)
    MaxwellBoltzmannDistribution(atoms, 300 * units.kB)
    calc = Amp.load("amp.amp")
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    atoms.get_kinetic_energy()
    atoms.get_forces()
    traj.write(atoms)

    dyn = VelocityVerlet(atoms, 5 * units.fs)
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


filename = "test.traj"
test_data(10000, 10, filename=filename)
