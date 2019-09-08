import sys
from amp.utilities import Annealer
from amp.descriptor.cutoffs import Cosine, Polynomial

sys.path.insert(1, "../tools")

from parameter_search import ParameterSearch
from training import Trainer


if __name__ == "__main__":
    pms = ParameterSearch()
    pms.create_train_test()
    trn = pms.create_trainer()

    rcs = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    gamma = 5.0
    cutoffs = []
    for rc in rcs:
        cutoffs.append(Cosine(rc))
        cutoffs.append(Polynomial(rc, gamma=gamma))

    calcs = {}
    for cutoff in cutoffs:
        trn.cutoff = cutoff
        trn.create_Gs(
            elements, num_radial_etas, num_angular_etas, num_zetas, angular_type
        )

        label = "{}-{}".format(cutoff.__class__.__name__, cutoff.Rc)
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

    columns = ["Cutoff", "Energy RMSE", "Force RMSE"]
    trn.test_calculators(calcs, test_traj, columns)
