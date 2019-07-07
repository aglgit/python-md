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
            "argon": self.argon_system,
            "silicon": self.silicon_system,
            "copper": self.copper_system,
        }

    def generate_system(self, calc, system, size, temp):
        if system in self.systems.keys():
            self.systems[system](size, temp)
            MaxwellBoltzmannDistribution(self.atoms, temp * units.kB)
            Stationary(self.atoms)
            ZeroRotation(self.atoms)
            self.atoms.set_calculator(calc)
        else:
            print("System {} not found!".format(system))

    def argon_system(self, size, temp):
        self.atoms = FaceCenteredCubic(size=size, symbol="Ar", pbc=True)

    def silicon_system(self, size, temp):
        self.atoms = Diamond(size=size, symbol="Si", pbc=True)

    def copper_system(self, size, temp):
        self.atoms = FaceCenteredCubic(size=size, symbol="Cu", pbc=True)

    def create_traj(self, filename, n_steps, save_interval, timestep=1.0):
        if os.path.exists(filename):
            print("File {} already exists!".format(filename))
            return

        print("Generating traj {}".format(filename))
        traj = ase.io.Trajectory(filename, "w")

        self.atoms.get_potential_energy()
        self.atoms.get_kinetic_energy()
        self.atoms.get_forces()
        energy = self.atoms.get_total_energy()
        traj.write(self.atoms)

        self.timestep = 0
        print("Timestep: {}, total energy: {}".format(self.timestep, energy))

        dyn = VelocityVerlet(self.atoms, timestep=timestep * units.fs)
        count = n_steps // save_interval
        for i in range(count):
            dyn.run(save_interval)
            self.atoms.get_potential_energy()
            self.atoms.get_kinetic_energy()
            self.atoms.get_forces()
            energy = self.atoms.get_total_energy()
            traj.write(self.atoms)
            self.timestep += save_interval
            print("Timestep: {}, total energy: {}".format(self.timestep, energy))

        print("Finished generating traj {}".format(filename))

    def continue_traj(
        self, filename, new_filename, n_steps, save_interval, timestep=1.0
    ):
        if os.path.exists(new_filename):
            print("File {} already exists!".format(new_filename))
            return

        print("Continuing traj {}".format(filename))
        traj = ase.io.read(filename, ":")
        self.atoms = traj[-1]
        traj = ase.io.Trajectory(new_filename, "w")

        self.atoms.get_potential_energy()
        self.atoms.get_kinetic_energy()
        self.atoms.get_forces()
        energy = self.atoms.get_total_energy()
        traj.write(self.atoms)

        print("Timestep: {}, total energy: {}".format(self.timestep, energy))

        dyn = VelocityVerlet(self.atoms, timestep=timestep * units.fs)
        count = n_steps // save_interval
        for i in range(count):
            dyn.run(save_interval)
            self.atoms.get_potential_energy()
            self.atoms.get_kinetic_energy()
            self.atoms.get_forces()
            energy = self.atoms.get_total_energy()
            traj.write(self.atoms)
            self.timestep += save_interval
            print("Timestep: {}, total energy: {}".format(self.timestep, energy))

        print("Finished generating traj {}".format(new_filename))

    def convert_traj(self, traj_file):
        xyz_file = "".join((traj_file.split(".")[0], ".xyz"))
        if not os.path.exists(traj_file):
            print("No such file {}!".format(traj_file))
        elif os.path.exists(xyz_file):
            print("File {} already exists!".format(xyz_file))
        else:
            print("Converting {} to {}".format(traj_file, xyz_file))
            traj = ase.io.read(traj_file, ":")
            ase.io.write(xyz_file, traj)
