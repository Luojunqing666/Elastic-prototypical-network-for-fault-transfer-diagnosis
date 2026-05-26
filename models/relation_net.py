"""Relation Network for few-shot fault diagnosis."""

import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Standard conv block: Conv1d -> BN -> ReLU -> MaxPool."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class EncoderNet(nn.Module):
    """
    Feature encoder for Relation Network.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        num_blocks: Number of convolutional blocks.
    """

    def __init__(self, in_channels=1, hidden_channels=64, num_blocks=4):
        super().__init__()
        layers = [conv_block(in_channels, hidden_channels)]
        layers += [conv_block(hidden_channels, hidden_channels) for _ in range(num_blocks - 1)]
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class RelationNet(nn.Module):
    """
    Relation module that computes similarity scores.

    Args:
        hidden_channels: Number of hidden channels (input is 2x due to concatenation).
        embed_size: Flattened feature size after conv blocks.
        hidden_size: Hidden layer size in the FC head.
    """

    def __init__(self, hidden_channels=64, embed_size=256, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            conv_block(hidden_channels * 2, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            nn.Flatten(),
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Compute relation score.

        Args:
            x: Concatenated support-query pairs [N, 2*hidden_channels, length].

        Returns:
            Relation scores [N, 1].
        """
        return self.network(x)
