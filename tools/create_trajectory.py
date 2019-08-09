import os
from ase import units
from ase.io import read, write, Trajectory
from ase.md.verlet import VelocityVerlet
from build_atoms import AtomBuilder


class TrajectoryBuilder:
    def __init__(self):
        self.atmb = AtomBuilder()

    def build_atoms(self, system, size, temp, calc, seed=None):
        atoms = self.atmb.build_atoms(system, size, temp, seed)
        atoms.set_calculator(calc)

        return atoms

    def integrate_atoms(
        self,
        atoms,
        traj_file,
        n_steps,
        save_interval,
        steps=0,
        timestep=5.0,
        traj_dir="trajs",
        convert=False,
    ):
        if not os.path.exists(traj_dir):
            os.mkdir(traj_dir)
        traj_file = os.path.join(traj_dir, traj_file)
        if not os.path.exists(traj_file):
            traj = Trajectory(traj_file, "w")
            print("Creating trajectory {}...".format(traj_file))

            dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)
            count = n_steps // save_interval
            for i in range(count):
                dyn.run(save_interval)
                energy = atoms.get_total_energy()
                forces = atoms.get_forces()
                traj.write(atoms)
                steps += save_interval
                print("Steps: {}, total energy: {}".format(steps, energy))
        else:
            print("Trajectory {} already exists!".format(traj_file))

        if convert:
            self.convert_trajectory(traj_file)

        return steps, traj_file

    def convert_trajectory(self, traj_file):
        xyz_file = "".join((traj_file.split(".")[0], ".xyz"))
        if not os.path.exists(traj_file):
            print("No such file {}!".format(traj_file))
        elif os.path.exists(xyz_file):
            print("File {} already exists!".format(xyz_file))
        else:
            print("Converting {} to {}...".format(traj_file, xyz_file))
            traj = read(traj_file, ":")
            write(xyz_file, traj)
