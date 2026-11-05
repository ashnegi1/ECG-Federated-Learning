"""Compact one-dimensional CNN for fixed-length ECG beats."""
import torch
from torch import nn


class ECGCNN(nn.Module):
    def __init__(self, input_length: int = 252, num_classes: int = 5, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected ECG tensor of shape (batch, length) or (batch, 1, length)")
        return self.classifier(self.features(x))
