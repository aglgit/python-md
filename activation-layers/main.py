import sys
from amp.utilities import Annealer

sys.path.insert(1, "../tools")

from parameter_search import ParameterSearch

if __name__ == "__main__":
    pms = ParameterSearch()
    pms.create_train_test()
    trn = pms.create_trainer()

    activation = ["tanh", "sigmoid"]
    hidden_layers = [[10], [20], [30], [40], [10, 10], [20, 10], [30, 10], [40, 40]]
    dblabel = "amp-train"
    calcs = {}
    for ac in activation:
        for hl in hidden_layers:
            trn.activation = ac
            trn.hidden_layers = hl
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

    columns = ["Activation/Hidden layers", "Energy RMSE", "Force RMSE"]
    dblabel = "amp-test"
    trn.test_calculators(calcs, test_traj, columns, dblabel=dblabel)
