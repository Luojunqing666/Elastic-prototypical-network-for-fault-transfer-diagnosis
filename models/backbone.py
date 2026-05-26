"""CNN backbone with optional Squeeze-and-Excitation (SE) attention."""

import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention module.

    Args:
        channel: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv1d -> BN -> ReLU -> MaxPool, with optional SE.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        pool_stride: MaxPool stride (controls downsampling factor).
        use_se: Whether to apply SE attention.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pool_stride=2, use_se=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride, ceil_mode=False)
        self.se = SELayer(out_channels) if use_se else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.se is not None:
            x = self.se(x)
        x = self.pool(x)
        return x


class CNN4Backbone(nn.Module):
    """
    4-layer 1D CNN backbone for vibration signal feature extraction.

    The SE attention is applied to the last convolutional block by default,
    following the EProtoNet paper design.

    Args:
        in_channels: Number of input channels (default: 1 for single-channel vibration).
        hidden_channels: Number of hidden channels in each conv block.
        num_layers: Number of convolutional blocks.
        pool_stride: MaxPool stride per block.
        use_se: Whether to apply SE attention on the last block.
    """

    def __init__(self, in_channels=1, hidden_channels=64, num_layers=4, pool_stride=2, use_se=True):
        super().__init__()
        layers = [ConvBlock(in_channels, hidden_channels, pool_stride=pool_stride)]
        for i in range(num_layers - 1):
            # Apply SE only on the last block
            se = use_se and (i == num_layers - 2)
            layers.append(ConvBlock(hidden_channels, hidden_channels, pool_stride=pool_stride, use_se=se))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        return x

    @property
    def output_dim(self):
        """Compute output dimension for a given input (lazy, run a dummy forward)."""
        return None  # Computed dynamically based on input size
