import sys
from gpaw import GPAW
from gpaw.tddft import TDDFT
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from ase.units import Hartree, Bohr, AUT
from ase.io import read, write, Trajectory

sys.path.insert(1, "../tools")

from build_atoms import AtomBuilder

if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 300

    atmb = AtomBuilder()
    atoms = atmb.build_atoms(system, size, temp)

    name = "test"
    calc = GPAW()
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write(name + ".gpw", mode="all")

    # Ehrenfest simulation parameters
    timestep = 10.0  # timestep given in attoseconds
    ndiv = 10  # write trajectory every 10 timesteps
    niter = 500  # run for 500 timesteps

    tdcalc = TDDFT(
        name + ".gpw", txt=name + "_td.txt", propagator="EFSICN", solver="BiCGStab"
    )

    ehrenfest = EhrenfestVelocityVerlet(tdcalc)
    traj = Trajectory(name + ".traj", "w", tdcalc.get_atoms())

    for i in range(100):
        ehrenfest.propagate(timestep)
        atoms.get_potential_energy()
        traj.write(atoms)
