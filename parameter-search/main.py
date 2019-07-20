import sys

sys.path.insert(0, "../tools")

import os
from asap3 import EMT
from amp import Amp
from amp.descriptor.cutoffs import Cosine, Polynomial
from build_atoms import AtomBuilder
from create_traj import CreateTrajectory
from train_amp import Trainer


if __name__ == "__main__":
    system = "copper"
    size = (2, 2, 2)
    temp = 300

    traj_dir = "trajs"
    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    train_traj = os.path.join(traj_dir, "training.traj")
    n_train = int(4e5)
    save_interval = 100

    if not os.path.exists(train_traj):
        atmb = AtomBuilder()
        atoms = atmb.build_atoms(system, size, temp)
        calc = EMT()
        atoms.set_calculator(calc)

        ctrj = CreateTrajectory()
        print("Creating trajectory {}".format(train_traj))
        ctrj.integrate_atoms(atoms, train_traj, n_train, save_interval)

    logfile = "loss.txt"
    parameters = {}
    parameters["force_coefficient"] = [
        1e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
        7.5e-2,
        1e-1,
        2e-1,
        3e-1,
        5e-1,
    ]
    parameters["hidden_layers"] = [
        (10),
        (20),
        (30),
        (40),
        (5, 5),
        (10, 10),
        (20, 10),
        (30, 10),
        (20, 20),
        (40, 40),
    ]
    parameters["activation"] = ["tanh", "sigmoid"]
    parameters["cutoff"] = [
        Cosine(3.0),
        Cosine(4.0),
        Cosine(5.0),
        Cosine(6.0),
        Cosine(7.0),
        Polynomial(3.0),
        Polynomial(4.0),
        Polynomial(5.0),
        Polynomial(6.0),
        Polynomial(7.0),
    ]

    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": int(1e3)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)
    Gs = None

    if not os.path.exists(logfile):
        log = open(logfile, "w")
        string = """
                 Convergence:        {}
                 Energy coefficient: {}
                 Force coefficient:  {}
                 Hidden layers:      {}
                 Activation:         {}
                 Cutoff:             {}
                 Gs:                 AMP defaults\n""".format(
            convergence,
            energy_coefficient,
            force_coefficient,
            hidden_layers,
            activation,
            cutoff,
        )
        log.write("Default parameters:\n")
        log.write(string)
        trn = Trainer(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            activation=activation,
            cutoff=cutoff,
            Gs=Gs,
        )
        amp_calc = trn.create_calc()
        loss, energy_rmse, force_rmse = amp_calc.train_return_loss(train_traj)

        string = "Parameter: {}, Value: {}, Loss: {:.3E}, Energy RMSE: {:.3E}, Force RMSE: {:.3E}\n".format(
            parameter, value, loss, energy_rmse, force_rmse
        )
        print(string, end="")
        log.write(string)

    else:
        log = open(logfile, "a")

    for parameter in parameters.keys():
        trn = Trainer(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            activation=activation,
            cutoff=cutoff,
            Gs=Gs,
        )
        values = parameters[parameter]
        log.write("\n")
        for value in values:
            setattr(trn, parameter, value)
            amp_calc = trn.create_calc()
            loss, energy_rmse, force_rmse = amp_calc.train_return_loss(train_traj)

            string = "Parameter: {}, Value: {}, Loss: {:.3E}, Energy RMSE: {:.3E}, Force RMSE: {:.3E}\n".format(
                parameter, value, loss, energy_rmse, force_rmse
            )
            print(string, end="")
            log.write(string)

    log.close()
