import os
import ase.io
from ase.lattice.cubic import Diamond
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from asap3 import OpenKIMcalculator


class GenerateTrajectory:
    def __init__(self, calc):
        self.calc = calc

    def stillinger_weber_system(self, size, temp):
        self.atoms = Diamond(size=size, symbol="Si", pbc=True)
        MaxwellBoltzmannDistribution(self.atoms, temp * units.kB)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        self.atoms.set_calculator(self.calc)

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
