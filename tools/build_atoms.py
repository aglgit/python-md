import numpy as np
from ase.lattice.cubic import FaceCenteredCubic, Diamond
from ase import units
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)


class AtomBuilder:
    def __init__(self):
        self.systems = {
            "argon": self.argon_system,
            "copper": self.copper_system,
            "silicon": self.silicon_system,
        }

    def build_atoms(self, system, size, temp, seed=None):
        assert system in self.systems.keys(), "System {} not found!".format(system)
        atoms = self.systems[system](size)
        if seed is not None:
            np.random.seed(seed)
        MaxwellBoltzmannDistribution(atoms, temp * units.kB, rng=np.random)
        Stationary(atoms)
        ZeroRotation(atoms)

        return atoms

    def argon_system(self, size):
        return FaceCenteredCubic(size=size, symbol="Ar", pbc=True)

    def copper_system(self, size):
        return FaceCenteredCubic(size=size, symbol="Cu", pbc=True)

    def silicon_system(self, size):
        return Diamond(size=size, symbol="Si", pbc=True)
