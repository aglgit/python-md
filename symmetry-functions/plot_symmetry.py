import sys

sys.path.insert(0, "../tools")
 
import numpy as np
import matplotlib.pyplot as plt
from analysis import Analyzer


def cosine(rij, rc):
    term = 0.5 * (1 + np.cos(np.pi * rij / rc))

    return term


def polynomial(rij, rc, gamma=4.0):
    term1 = 1 + gamma * (rij / rc) ** (gamma + 1)
    term2 = (gamma + 1) * (rij / rc) ** gamma

    return term1 + term2


def G2_func(eta, rij, rc, cutoff=cosine):
    term1 = np.exp(-eta * (rij / rc) ** 2)
    term2 = cosine(rij, rc)

    return term1 * term2


def G4_func(gamma, zeta, theta_ijk):
    term1 = 2 ** (1 - zeta)
    term2 = (1 + gamma * np.cos(theta_ijk)) ** zeta

    return term1 * term2


if __name__ == "__main__":

    test = {
        "Gs-4-4-G4": [
            {"type": "G2", "element": "Cu", "eta": 0.1},
            {"type": "G2", "element": "Cu", "eta": 1.0},
            {"type": "G2", "element": "Cu", "eta": 2.5},
            {"type": "G2", "element": "Cu", "eta": 5.0},
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": 1.0,
                "zeta": 1,
            },
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": -1.0,
                "zeta": 1,
            },
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": 1.0,
                "zeta": 2,
            },
            {
                "type": "G4",
                "elements": ["Cu", "Cu"],
                "eta": 0.001,
                "gamma": -1.0,
                "zeta": 2,
            },
        ]
    }

    anl = Analyzer(save_interval=100)
    train_traj = "trajs/training.traj"
    rc = 6.0
    r, rdf = anl.calculate_rdf(train_traj, rmax=rc)
    rdf[np.nonzero(rdf)] /= max(rdf)
    print(rdf)
    plt.plot(r, rdf)
    theta = np.linspace(0, np.pi, 10000)
    for Gs in test:
        for i in range(len(test[Gs])):
            symm_func = test[Gs][i]
            if symm_func["type"] == "G2":
                eta = symm_func["eta"]
                y = G2_func(eta, r, rc)
                plt.plot(r, y, label="eta={}".format(eta))
            elif symm_func["type"] in ["G4", "G5"]:
                pass
#                gamma = symm_func["gamma"]
#                zeta = symm_func["zeta"]
#                y = G4_func(gamma, zeta, theta)

    plt.legend()
    plt.show()
