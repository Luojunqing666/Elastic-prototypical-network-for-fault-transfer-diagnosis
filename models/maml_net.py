"""MAML-compatible CNN for few-shot fault diagnosis."""

import torch
import torch.nn as nn


def maml_init_(module):
    """Xavier initialization for MAML compatibility."""
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class MAMLConvBlock(nn.Module):
    """Conv block with MAML-compatible initialization."""

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)
        maml_init_(self.conv)
        self.bn = nn.BatchNorm1d(out_channels, affine=True)
        nn.init.uniform_(self.bn.weight)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride, ceil_mode=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MAMLNet(nn.Module):
    """
    MAML-compatible CNN for few-shot classification.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        num_layers: Number of convolutional blocks.
        sample_len: Input signal length (for computing embedding size).
    """

    def __init__(self, num_classes=5, in_channels=1, hidden_channels=64,
                 num_layers=4, sample_len=1024):
        super().__init__()
        layers = [MAMLConvBlock(in_channels, hidden_channels)]
        for _ in range(num_layers - 1):
            layers.append(MAMLConvBlock(hidden_channels, hidden_channels))
        self.features = nn.Sequential(*layers)

        # Compute embedding size: signal_len / (pool_stride ^ num_layers) * hidden_channels
        embed_size = (sample_len // (2 ** num_layers)) * hidden_channels
        self.classifier = nn.Linear(embed_size, num_classes, bias=True)
        maml_init_(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
