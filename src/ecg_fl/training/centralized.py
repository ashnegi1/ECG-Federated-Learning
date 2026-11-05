"""Centralized CNN baseline using precisely the same training records and architecture."""
from pathlib import Path
import torch
import numpy as np

from ecg_fl.models.cnn import ECGCNN
from ecg_fl.training.engine import train_model, evaluate_loss


def run_centralized(x, y, *, validation_data=None, epochs=10, batch_size=128, learning_rate=1e-3,
                    class_weighting=False, class_weights=None,
                    output_path="outputs/centralized_model.pt", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ECGCNN(input_length=x.shape[1])
    train_validation = None if validation_data is None else (validation_data, True)
    if train_validation is None:
        losses = train_model(model, x, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                             class_weighting=class_weighting, class_weights=class_weights)
        train_losses, validation_losses = losses, []
    else:
        losses = train_model(model, x, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                             class_weighting=class_weighting, class_weights=class_weights,
                             validation_data=validation_data, return_validation=True)
        train_losses, validation_losses = losses
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    validation_loss = None if not validation_losses else validation_losses[-1]
    return model, {"train_loss_by_epoch": train_losses, "validation_loss_by_epoch": validation_losses,
                   "validation_loss": validation_loss, "class_weighting": bool(class_weighting),
                   "class_weights": None if not class_weighting else list(map(float, class_weights))}
