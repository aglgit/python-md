import ase.io
import numpy as np
from asap3.analysis.rdf import RadialDistributionFunction


class Analyzer:
    def __init__(self, save_interval):
        self.save_interval = save_interval

    def calculate_rdf(self, traj_file, rmax=10.0, nbins=100):
        traj = ase.io.read(traj_file, ":")
        x = (np.arange(nbins) + 0.5) * rmax / nbins
        rdf_obj = None
        for atoms in traj:
            if rdf_obj is None:
                rdf_obj = RadialDistributionFunction(atoms, rmax, nbins)
            else:
                rdf_obj.atoms = atoms
            rdf_obj.update()
        rdf = rdf_obj.get_rdf()

        return x, rdf

    def calculate_msd(self, traj_file):
        traj = ase.io.read(traj_file, ":")

        steps = np.arange(len(traj)) * self.save_interval
        msd = np.zeros(len(traj))
        init_pos = traj[0].get_positions()
        for i, atoms in enumerate(traj[1:]):
            pos = atoms.get_positions()
            disp = pos - init_pos
            msd[i + 1] = msd[i - 1] + np.linalg.norm(disp, axis=1).sum()

        return steps, msd

    def calculate_energy_diff(self, test_traj_file, amp_traj_file):
        test_traj = ase.io.read(test_traj_file, ":")
        amp_traj = ase.io.read(amp_traj_file, ":")

        num_images = len(test_traj)
        steps = np.arange(num_images) * self.save_interval
        energy_exact = np.zeros(num_images)
        energy_amp = np.zeros(num_images)
        for i in range(num_images):
            energy_exact[i] = test_traj[i].get_potential_energy()
            energy_amp[i] = amp_traj[i].get_potential_energy()

        return steps, energy_exact, energy_amp
