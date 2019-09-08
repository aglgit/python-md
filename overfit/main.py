import sys
from amp.utilities import Annealer

sys.path.insert(1, "../tools")

from parameter_search import ParameterSearch

if __name__ == "__main__":
    pms = ParameterSearch()
    pms.create_train_test()
    trn = pms.create_trainer()

    overfit = [10**(-i) for i in range(7)]
    dblabel = "amp-train"
    calcs = {}
    for of in overfit:
        trn.overfit = of
        label = "{}-{}".format(ac, hl)
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

    columns = ["Regularization", "Energy RMSE", "Force RMSE"]
    dblabel = "amp-test"
    trn.test_calculators(calcs, test_traj, columns, dblabel=dblabel)
