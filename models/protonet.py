"""
Elastic Prototypical Network (EProtoNet) for few-shot fault diagnosis.

Key contribution: Introduces an elasticity factor (scale_factor) to the
Euclidean distance metric, enabling adaptive distance scaling for better
prototype-based classification under domain shift.
"""

import torch
import torch.nn as nn
from .backbone import CNN4Backbone


class EProtoNet(nn.Module):
    """
    Elastic Prototypical Network.

    Uses a CNN backbone to extract embeddings, computes class prototypes
    from support samples, and classifies query samples using scaled
    Euclidean distance.

    Args:
        in_channels: Number of input signal channels.
        hidden_channels: Hidden dimension of CNN backbone.
        num_layers: Number of CNN blocks.
        use_se: Whether to use SE attention in backbone.
        scale_factor: Elasticity factor for distance scaling (default: 0.01).
    """

    def __init__(self, in_channels=1, hidden_channels=64, num_layers=4,
                 use_se=True, scale_factor=0.01):
        super().__init__()
        self.backbone = CNN4Backbone(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            use_se=use_se,
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        """Extract feature embeddings."""
        return self.backbone(x)

    @staticmethod
    def compute_prototypes(support_embeddings, ways, shots):
        """
        Compute class prototypes by averaging support embeddings.

        Args:
            support_embeddings: [ways * shots, embed_dim]
            ways: Number of classes.
            shots: Number of support samples per class.

        Returns:
            Prototypes tensor [ways, embed_dim].
        """
        return support_embeddings.reshape(ways, shots, -1).mean(dim=1)

    def scaled_euclidean_distance(self, query, prototypes):
        """
        Compute scaled Euclidean distance (negative logits).

        This is the core of the "elastic" mechanism: the distance is divided
        by a scale_factor to control the sharpness of the distance distribution.

        Args:
            query: Query embeddings [n_query, embed_dim].
            prototypes: Class prototypes [ways, embed_dim].

        Returns:
            Negative scaled distances (logits) [n_query, ways].
        """
        query = query.unsqueeze(1)       # [n_query, 1, embed_dim]
        prototypes = prototypes.unsqueeze(0)  # [1, ways, embed_dim]
        distances = torch.pow(query - prototypes, 2).sum(dim=-1)  # [n_query, ways]
        return -distances / self.scale_factor

    def euclidean_distance(self, query, prototypes):
        """
        Standard Euclidean distance (without elasticity factor).

        Args:
            query: Query embeddings [n_query, embed_dim].
            prototypes: Class prototypes [ways, embed_dim].

        Returns:
            Negative distances (logits) [n_query, ways].
        """
        query = query.unsqueeze(1)
        prototypes = prototypes.unsqueeze(0)
        return -torch.pow(query - prototypes, 2).mean(dim=-1)
