import sys
from gpaw import GPAW, mpi
from gpaw.tddft import TDDFT
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from ase.units import Hartree, Bohr, AUT
from ase.io import read, write, Trajectory

sys.path.insert(1, "../tools")

from build_atoms import AtomBuilder

if __name__ == "__main__":
    system = "copper"
    size = (1, 1, 1)
    temp = 500

    rank = mpi.world.rank

    atmb = AtomBuilder()
    atoms = atmb.build_atoms(system, size, temp)

    calc = GPAW(symmetry={'point_group': False})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write(system + ".gpw", mode="all")

    # Ehrenfest simulation parameters
    timestep = 10.0     # timestep given in attoseconds
    count = 5000        # run for 500 timesteps

    tdcalc = TDDFT(
        system + ".gpw", txt=system + "_td.txt", propagator="EFSICN", solver="BiCGStab"
    )
    ehrenfest = EhrenfestVelocityVerlet(tdcalc)
    if rank == 0:
        traj = Trajectory("training.traj", "w", tdcalc.get_atoms())

    for i in range(count):
        if rank == 0:
            energy = tdcalc.get_td_energy() * Hartree
            f = ehrenfest.F * Hartree / Bohr
            v = ehrenfest.v * Bohr / AUT
            atoms = tdcalc.atoms.copy()
            atoms.set_velocities(v)
            print("Step: {}, Total energy: {}".format(i, energy))
            traj.write(atoms, energy=energy, forces=f)

        ehrenfest.propagate(timestep)
