import sys
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial

sys.path.insert(1, "../tools")

from parameter_search import ParameterSearch
from training import Trainer


if __name__ == "__main__":
    pms = ParameterSearch()
    pms.create_train_test()

    max_steps = int(2e3)
    convergence = {"energy_rmse": 1e-16, "force_rmse": None, "max_steps": max_steps}
    force_coefficient = None
    cutoff = Polynomial(6.0, gamma=5.0)
    trn = Trainer(
        convergence=convergence, force_coefficient=force_coefficient, cutoff=cutoff
    )

    num_radial_etas = [4, 5, 6, 7, 8, 9, 10, 10]
    num_angular_etas = [n + 2 for n in num_radial_etas]
    num_zetas = [1, 1, 1, 1, 1, 1, 1, 2]
    gammas = [1.0, -1.0]
    symm_funcs = {"Default": None}
    for i in range(len(num_radial_etas)):
        for angular_type in ["G4", "G5"]:
            nr = num_radial_etas[i]
            na = num_angular_etas[i]
            nz = num_zetas[i]

            trn.create_Gs(elements, nr, na, nz, angular_type)
            label = "Gs-{}-{}-{}-{}".format(nr, na, nz, angular_type)
            symm_funcs[label] = trn.Gs

    calcs = {}
    for label, symm_func in symm_funcs.items():
        trn = Trainer(
            convergence=convergence,
            force_coefficient=force_coefficient,
            cutoff=cutoff,
            Gs=symm_func,
        )
        dblabel = label + "-train"
        calc = trn.create_calc(label=label, dblabel=dblabel)
        ann = Annealer(
            calc=calc,
            images=train_traj,
            Tmax=20,
            Tmin=1,
            steps=2000,
            train_forces=False,
        )
        amp_name = trn.train_calc(calc, train_traj)
        calcs[label] = amp_name

    columns = ["Symmetry function", "Energy RMSE", "Force RMSE"]
    trn.test_calculators(calcs, test_traj, columns)
