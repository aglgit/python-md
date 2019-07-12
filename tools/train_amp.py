import os
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction


class Trainer:
    def __init__(
        self,
        convergence=None,
        energy_coefficient=1.0,
        force_coefficient=None,
        hidden_layers=(10),
        activation="tanh",
        cutoff=6.0,
        Gs=None,
    ):
        if convergence is None:
            self.convergence = {"energy_rmse": 1e-3, "max_steps": int(1e3)}
        else:
            self.convergence = convergence

        self.energy_coefficient = energy_coefficient
        self.force_coefficient = force_coefficient
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.cutoff = cutoff
        self.Gs = Gs

    def create_calc(self):
        descriptor = Gaussian(cutoff=self.cutoff, Gs=self.Gs, fortran=True)
        loss_function = LossFunction(
            convergence=self.convergence,
            energy_coefficient=self.energy_coefficient,
            force_coefficient=self.force_coefficient,
        )
        model = NeuralNetwork(
            hiddenlayers=self.hidden_layers,
            activation=self.activation,
            lossfunction=loss_function,
        )
        calc = Amp(descriptor=descriptor, model=model)

        return calc
