import os
import ase.io
from ase.lattice.cubic import Diamond
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.calculators.cp2k import CP2K

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class GenerateTrajectory:
    def __init__(self, calc):
        self.calc = calc

    def stillinger_weber_system(self, size, temp):
        self.atoms = Diamond(size=size, symbol="Si", pbc=True)
        MaxwellBoltzmannDistribution(self.atoms, temp * units.kB)
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


if __name__ == "__main__":
    train_filename = "training.traj"
    n_steps = 100
    save_interval = 10
    size = (2, 2, 2)
    temp = 300
    calc = CP2K()

    cp2k_train = GenerateTrajectory(calc)
    cp2k_train.stillinger_weber_system(size, temp)
    cp2k_train.create_traj(train_filename, n_steps, save_interval)

    print("Training from traj: {}".format(train_filename))
    traj = ase.io.read(train_filename, ":")
    descriptor = Gaussian(cutoff=6.0, fortran=True)
    convergence = {"energy_rmse": 1e-3, "force_rmse": 5e-2}
    loss_function = LossFunction(convergence=convergence)
    model = NeuralNetwork(activation="tanh", lossfunction=loss_function)

    calc = Amp(descriptor=descriptor, model=model)
    calc.train(images=traj)
