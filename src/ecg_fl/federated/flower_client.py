"""Flower NumPyClient: local ECG training; only parameters and scalars are returned."""
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl

from ecg_fl.models.cnn import ECGCNN
from ecg_fl.training.engine import train_model, evaluate_loss, choose_device


def get_parameters(model: torch.nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy().copy() for _, value in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    state = OrderedDict((key, torch.as_tensor(value)) for key, value in zip(model.state_dict(), parameters))
    model.load_state_dict(state, strict=True)


class ECGFlowerClient(fl.client.NumPyClient):
    def __init__(self, x: np.ndarray, y: np.ndarray, config: dict):
        self.x, self.y, self.config = x, y, config
        self.model = ECGCNN(input_length=x.shape[1], dropout=config.get("dropout", 0.25))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        losses = train_model(
            self.model, self.x, self.y,
            epochs=int(config.get("local_epochs", self.config.get("local_epochs", 1))),
            batch_size=int(self.config.get("batch_size", 128)),
            learning_rate=float(self.config.get("learning_rate", 1e-3)),
            class_weighting=bool(self.config.get("class_weighting", False)),
            class_weights=self.config.get("class_weights"),
            device=choose_device(),
        )
        # No examples, labels, waveforms, gradients, or patient identifiers leave this client.
        return get_parameters(self.model), len(self.y), {"train_loss": float(losses[-1])}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss = evaluate_loss(self.model, self.x, self.y)
        return loss, len(self.y), {"accuracy": float("nan")}
