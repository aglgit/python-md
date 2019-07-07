import os
import numpy as np
import matplotlib.pyplot as plt
import ase.io
from ase import Atom, Atoms, units
from ase.calculators.lj import LennardJones
from amp import Amp

if not os.path.exists("amp.amp"):
    print("No trained AMP calculator!")
else:
    amp_calc = Amp.load("amp.amp")

epsilon = 1.0318e-2
sigma = 3.405
calc = LennardJones(sigma=sigma, epsilon=epsilon)
atoms = Atoms([Atom("Ar", (0, 0, 0)), Atom("Ar", (0, 0, 0))])
atoms.set_calculator(calc)

dist = np.linspace(0.95, 3.0, 1000) * sigma
pot = np.zeros_like(dist)

for i, r in enumerate(dist):
    atoms.positions[1][0] = r
    pot[i] = atoms.get_potential_energy()

plt.plot(dist, pot)
plt.show()
