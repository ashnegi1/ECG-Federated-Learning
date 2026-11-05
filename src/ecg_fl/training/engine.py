"""Shared PyTorch training and evaluation routines."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def choose_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(np.array(x, copy=True), dtype=torch.float32), torch.tensor(np.array(y, copy=True), dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def fedprox_penalty(
    model: nn.Module,
    global_parameters: Mapping[str, torch.Tensor] | None,
    mu: float,
) -> torch.Tensor:
    """Return FedProx's ``mu/2 * ||w - w_global||^2`` over trainable parameters."""
    if mu < 0 or not np.isfinite(mu):
        raise ValueError("proximal_mu must be finite and non-negative")
    parameters = dict(model.named_parameters())
    if mu == 0:
        return next(model.parameters()).new_zeros(())
    if global_parameters is None:
        raise ValueError("global_parameters are required when proximal_mu is positive")
    if parameters.keys() != global_parameters.keys():
        raise ValueError("global parameter names do not match the local model")
    terms = [
        (parameter - global_parameters[name].detach().to(parameter.device)).pow(2).sum()
        for name, parameter in parameters.items()
    ]
    if not terms:
        return torch.zeros(())
    return (float(mu) / 2.0) * torch.stack(terms).sum()


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weighting: bool = False,
    class_weights: np.ndarray | list[float] | None = None,
    device: torch.device | None = None,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    return_validation: bool = False,
    proximal_mu: float = 0.0,
    proximal_reference: Mapping[str, torch.Tensor] | None = None,
) -> list[float] | tuple[list[float], list[float]]:
    device = device or choose_device()
    model.to(device).train()
    if proximal_mu < 0 or not np.isfinite(proximal_mu):
        raise ValueError("proximal_mu must be finite and non-negative")
    if proximal_mu > 0 and proximal_reference is None:
        raise ValueError("proximal_reference is required when proximal_mu is positive")
    reference = (
        {name: value.detach().to(device).clone() for name, value in proximal_reference.items()}
        if proximal_mu > 0 and proximal_reference is not None else None
    )
    weights = None
    if class_weighting:
        if class_weights is None:
            raise ValueError("class_weights must be derived from the global training split")
        weight_array = np.asarray(class_weights, dtype=np.float32)
        if weight_array.shape != (5,) or not np.isfinite(weight_array).all() or np.any(weight_array < 0):
            raise ValueError("class_weights must contain five finite, non-negative values")
        weights = torch.as_tensor(weight_array, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = make_loader(x, y, batch_size, True)
    losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(epochs):
        total_loss, seen = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            if proximal_mu > 0:
                loss = loss + fedprox_penalty(model, reference, proximal_mu)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            seen += len(by)
        losses.append(total_loss / max(seen, 1))
        if validation_data is not None:
            validation_losses.append(evaluate_loss(model, validation_data[0], validation_data[1], batch_size, device))
    return (losses, validation_losses) if return_validation else losses


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512, device: torch.device | None = None) -> np.ndarray:
    device = device or choose_device()
    model.to(device).eval()
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, False)
    outputs = []
    for bx, _ in loader:
        outputs.append(model(bx.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.int64)


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int = 512, device=None) -> float:
    device = device or choose_device()
    model.to(device).eval()
    loader = make_loader(x, y, batch_size, False)
    total, count = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            total += criterion(model(bx), by).item() * len(by)
            count += len(by)
    return total / max(count, 1)
