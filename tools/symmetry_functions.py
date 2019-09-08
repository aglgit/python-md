import numpy as np

def cosine(rij, r_cut):
    term = 0.5 * (1 + np.cos(np.pi * rij / r_cut))

    return term

def polynomial(rij, r_cut, gamma=4.0):
    term1 = 1 + gamma * (rij / r_cut) ** (gamma + 1)
    term2 = (gamma + 1) * (rij / r_cut) ** gamma

    return term1 - term2

def G2(eta, center, rij, cutoff, r_cut):
    term1 = np.exp(-eta * (rij - center) ** 2 / r_cut ** 2)
    term2 = cutoff(rij, r_cut)

    return term1 * term2

def G4(eta, gamma, zeta, theta_ijk):
    term1 = 2 ** (1 - zeta)
    term2 = (1 + gamma * np.cos(theta_ijk)) ** zeta
    term3 = np.exp(-eta)

    return term1 * term2 * term3

