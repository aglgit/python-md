import numpy as np
from ase.io import read
from asap3.analysis import CoordinationNumbers
from asap3.analysis.rdf import RadialDistributionFunction


class Analyzer:
    def __init__(self, save_interval=10):
        self.save_interval = save_interval

    def calculate_rdf(self, traj_file, rmax=10.0, nbins=100):
        traj = read(traj_file, ":")

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

    def calculate_coordination_number(self, traj_file, r_cut):
        traj = read(traj_file, ":")

        coord_min_max = np.zeros((len(traj), 2))
        coord_all_avg = np.zeros(len(traj))

        for i, atoms in enumerate(traj):
            c = CoordinationNumbers(atoms, rCut=r_cut)
            coord_min_max[i] = c.min(), c.max()
            coord_all_avg[i] = np.mean(c)

        coord_min = coord_min_max[:, 0].min()
        coord_max = coord_min_max[:, 1].max()
        coord_avg = np.mean(coord_all_avg)

        return coord_min, coord_max, coord_avg

    def calculate_msd(self, traj_file):
        traj = read(traj_file, ":")

        steps = np.arange(len(traj)) * self.save_interval
        msd = np.zeros(len(traj))
        init_pos = traj[0].get_positions()
        for i, atoms in enumerate(traj[1:]):
            pos = atoms.get_positions()
            disp = pos - init_pos
            msd[i + 1] = msd[i - 1] + np.linalg.norm(disp, axis=1).sum()

        return steps, msd

    def calculate_energy_diff(self, test_traj_file, amp_traj_file):
        test_traj = read(test_traj_file, ":")
        amp_traj = read(amp_traj_file, ":")

        num_images = len(test_traj)
        steps = np.arange(num_images) * self.save_interval
        energy_exact = np.zeros(num_images)
        energy_amp = np.zeros(num_images)
        for i in range(num_images):
            energy_exact[i] = test_traj[i].get_total_energy()
            energy_amp[i] = amp_traj[i].get_total_energy()

        return steps, energy_exact, energy_amp

    def calculate_pot_energy_diff(self, test_traj_file, amp_traj_file):
        test_traj = read(test_traj_file, ":")
        amp_traj = read(amp_traj_file, ":")

        num_images = len(test_traj)
        steps = np.arange(num_images) * self.save_interval
        energy_exact = np.zeros(num_images)
        energy_amp = np.zeros(num_images)
        for i in range(num_images):
            energy_exact[i] = test_traj[i].get_potential_energy()
            energy_amp[i] = amp_traj[i].get_potential_energy()

        return steps, energy_exact, energy_amp

    def calculate_amp_error(self, test_traj_file, calc):
        test_traj = read(test_traj_file, ":")

        num_images = len(test_traj)
        num_atoms = len(test_traj[0])
        energy_exact = np.zeros(num_images)
        energy_amp = np.zeros(num_images)
        force_exact = np.zeros((num_images, 3 * num_atoms))
        force_amp = np.zeros((num_images, 3 * num_atoms))
        energy_diff = np.zeros(num_images)
        force_diff = np.zeros((num_images, 3 * num_atoms))

        for i in range(num_images):
            energy_exact[i] = test_traj[i].get_potential_energy()
            force_exact[i] = test_traj[i].get_forces().reshape(-1)

            energy_amp[i] = calc.get_potential_energy(test_traj[i])
            force_amp[i] = calc.get_forces(test_traj[i]).reshape(-1)

            energy_diff[i] = abs(energy_exact[i] - energy_amp[i])
            force_diff[i] = abs(force_exact[i] - force_amp[i])

        force_exact = force_exact.reshape(-1)
        force_diff = force_diff.reshape(-1)

        return energy_exact, energy_diff, force_exact, force_diff
