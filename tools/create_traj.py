import os
import numpy as np
from ase import units
from ase.io import read, write, Trajectory
from ase.md.verlet import VelocityVerlet


class CreateTrajectory:
    def __init__(self, timestep=1.0):
        self.steps = 0
        self.timestep = timestep

    def integrate_atoms(self, atoms, traj_file, n_steps, save_interval):
        assert not os.path.exists(traj_file), "Trajectory {} already exists!".format(
            traj_file
        )
        traj = Trajectory(traj_file, "w")

        dyn = VelocityVerlet(atoms, timestep=self.timestep * units.fs)
        count = n_steps // save_interval
        for i in range(count):
            dyn.run(save_interval)
            energy = atoms.get_total_energy()
            forces = atoms.get_forces()
            traj.write(atoms)
            self.steps += save_interval
            print("Steps: {}, total energy: {}".format(self.steps, energy))

    def convert_trajectory(self, traj_file):
        xyz_file = "".join((traj_file.split(".")[0], ".xyz"))
        if not os.path.exists(traj_file):
            print("No such file {}!".format(traj_file))
        elif os.path.exists(xyz_file):
            print("File {} already exists!".format(xyz_file))
        else:
            print("Converting {} to {}".format(traj_file, xyz_file))
            traj = read(traj_file, ":")
            write(xyz_file, traj)
