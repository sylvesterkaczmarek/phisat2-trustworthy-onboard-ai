from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small convolutional classifier used only by the demonstration pipeline."""

    def __init__(self, in_ch: int = 3, num_classes: int = 2, base: int = 16):
        super().__init__()
        if in_ch <= 0 or num_classes < 2 or base <= 0:
            raise ValueError("in_ch, num_classes, and base must be positive")
        self.in_ch = int(in_ch)
        self.num_classes = int(num_classes)
        self.base = int(base)
        c1, c2, c3 = base, base * 2, base * 4
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))
