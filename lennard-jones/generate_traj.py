import os
import ase.io
from ase.lattice.cubic import FaceCenteredCubic, Diamond
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet


class GenerateTrajectory:
    def __init__(self):
        self.systems = {
            "lennard_jones": self.lennard_jones_system,
            "stillinger_weber": self.stillinger_weber_system,
        }

    def generate_system(self, calc, system, size, temp):
        if system in self.systems:
            self.systems[system](size, temp)
            MaxwellBoltzmannDistribution(self.atoms, temp * units.kB)
            Stationary(self.atoms)
            ZeroRotation(self.atoms)
            self.atoms.set_calculator(calc)
        else:
            print("System {} not found!".format(system))

    def lennard_jones_system(self, size, temp):
        self.atoms = FaceCenteredCubic(size=size, symbol="Ar", pbc=True)

    def stillinger_weber_system(self, size, temp):
        self.atoms = Diamond(size=size, symbol="Si", pbc=True)

    def create_traj(self, filename, n_steps, save_interval, timestep=5.0):
        if os.path.exists(filename):
            print("File {} already exists!".format(filename))
            return
        traj = ase.io.Trajectory(filename, "w")

        print("Generating traj {}".format(filename))
        self.atoms.get_potential_energy()
        self.atoms.get_kinetic_energy()
        self.atoms.get_forces()
        traj.write(self.atoms)
        print("Timestep: 0")

        dyn = VelocityVerlet(self.atoms, timestep=timestep * units.fs)
        count = n_steps // save_interval
        for i in range(count):
            dyn.run(save_interval)
            self.atoms.get_potential_energy()
            self.atoms.get_kinetic_energy()
            self.atoms.get_forces()
            traj.write(self.atoms)
            print("Timestep: {}".format((i + 1) * save_interval))

        print("Finished generating traj {}".format(filename))

    def convert_traj(self, infile, outfile):
        print("Converting {} to {}".format(infile, outfile))
        if not os.path.exists(infile):
            print("No such file {}!".format(infile))
        elif os.path.exists(outfile):
            print("File {} already exists!".format(outfile))
        else:
            traj = ase.io.read(infile, ":")
            ase.io.write(outfile, traj)
