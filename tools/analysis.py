import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
from asap3.analysis import CoordinationNumbers
from asap3.analysis.rdf import RadialDistributionFunction


class Analyzer:
    def __init__(self, save_interval=10):
        self.save_interval = save_interval

    def calculate_rdf(self, traj_file, r_max=10.0, nbins=100):
        traj = read(traj_file, ":")

        x = (np.arange(nbins) + 0.5) * r_max / nbins
        rdf_obj = None
        for atoms in traj:
            if rdf_obj is None:
                rdf_obj = RadialDistributionFunction(atoms, r_max, nbins)
            else:
                rdf_obj.atoms = atoms
            rdf_obj.update()
        rdf = rdf_obj.get_rdf()

        return x, rdf

    def calculate_adf(self, traj_file, r_cut, nbins=100):
        traj = read(traj_file, ":")

        theta_vals = np.linspace(0, 180, nbins)
        num_images = len(traj)
        num_atoms = len(traj[0])
        angles = np.zeros((num_images, num_atoms, 4*num_atoms, 4*num_atoms))
        nl = NeighborList(cutoffs=[r_cut / 2.0] * num_atoms,
                          self_interaction=False)
        for i, atoms in enumerate(traj):
            nl.update(atoms)
            cell = atoms.cell
            for j, atom in enumerate(atoms):
                selfposition = atom.position
                neighborindices, neighboroffsets = nl.get_neighbors(j)
                neighborpositions = atoms.positions[neighborindices] + np.dot(neighboroffsets, cell)
                dist = neighborpositions - selfposition
                norm = np.linalg.norm(dist, axis=1)
                for k, n1 in enumerate(neighborpositions):
                    rij = dist[k]
                    for l, n2 in enumerate(neighborpositions[k+1:]):
                        rik = dist[l]
                        cos_theta = np.dot(rij, rik) / (norm[k] * norm[l])
                        theta = np.arccos(cos_theta)
                        angles[i][j][k][l] = theta
        
        angles = angles.reshape(-1)
        angles = angles[~np.isnan(angles)]
        angles = angles[np.nonzero(angles)]
        angles *= 180 / np.pi
        adf, bin_edges = np.histogram(angles, bins=theta_vals, density=True)

        return bin_edges, adf

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
