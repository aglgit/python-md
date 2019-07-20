import sys

sys.path.insert(0, "../tools")

import os
from asap3 import EMT
from amp import Amp
from amp.descriptor.cutoffs import Polynomial
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
    elements = ["Cu"]
    Gs = []
    G2 = make_symmetry_functions(
        elements=elements,
        type="G2",
        etas=np.logspace(np.log10(0.01), np.log10(5.0), num=10),
    )
    for eta in [0.01, 0.05, 0.1]:
        G4 = make_symmetry_functions(
            elements=elements,
            type="G4",
            etas=[eta],
            zetas=[1.0, 2.0, 4.0, 16.0],
            gammas=[1.0, -1.0],
        )
        G5 = make_symmetry_functions(
            elements=elements,
            type="G5",
            etas=[eta],
            zetas=[1.0, 2.0, 4.0, 16.0],
            gammas=[1.0, -1.0],
        )
        Gs.append(G2 + G4)
        Gs.append(G2 + G5)

    convergence = {"energy_rmse": 1e-16, "force_rmse": 1e-16, "max_steps": int(1e3)}
    energy_coefficient = 1.0
    force_coefficient = 0.1
    hidden_layers = (10, 10)
    activation = "tanh"
    cutoff = Polynomial(6.0)

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
            Gs=None,
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

    for G in Gs:
        trn = Trainer(
            convergence=convergence,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
            hidden_layers=hidden_layers,
            activation=activation,
            cutoff=cutoff,
            Gs=G,
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
