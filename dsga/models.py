from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class ResNet10(nn.Module):
    """ResNet-10 encoder exposing Layer1..Layer4 for DSGA alignment."""

    feature_dims = (64, 128, 256, 512)

    def __init__(self, num_classes: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._layer(64, 64, stride=1)
        self.layer2 = self._layer(64, 128, stride=2)
        self.layer3 = self._layer(128, 256, stride=2)
        self.layer4 = self._layer(256, 512, stride=2)
        self.classifier = nn.Linear(512, num_classes)
        self._initialize()

    @staticmethod
    def _layer(in_channels: int, channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(BasicBlock(in_channels, channels, stride))

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

    def embeddings(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in self.feature_maps(x)]

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        embeddings = self.embeddings(x)
        if return_embeddings:
            return embeddings
        return self.classifier(embeddings[-1])


def build_model(num_classes: int, architecture: str = "resnet10") -> ResNet10:
    if architecture.lower() != "resnet10":
        raise ValueError(f"Unsupported architecture: {architecture}")
    return ResNet10(num_classes=num_classes)


def load_encoder_weights(model: nn.Module, state: dict, strict: bool = False) -> None:
    weights = state.get("model", state)
    if not strict:
        weights = {k: v for k, v in weights.items() if not k.startswith("classifier.")}
    model.load_state_dict(weights, strict=strict)

