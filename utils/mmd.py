"""MMD loss for domain adaptation."""

import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss with multi-kernel Gaussian.

    Reference:
        https://github.com/ZongxianLee/MMD_Loss.Pytorch

    Args:
        kernel_mul: Base multiplier for bandwidth.
        kernel_num: Number of Gaussian kernels.
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul

    def gaussian_kernel(self, source, target):
        """Compute multi-kernel Gaussian kernel matrix."""
        n_samples = source.shape[0] + target.shape[0]
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.shape[0], total.shape[0], total.shape[1])
        total1 = total.unsqueeze(1).expand(total.shape[0], total.shape[0], total.shape[1])
        L2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """
        Compute MMD loss between source and target distributions.

        Args:
            source: Source domain features [batch_size, feature_dim].
            target: Target domain features [batch_size, feature_dim].

        Returns:
            Scalar MMD loss.
        """
        batch_size = source.shape[0]
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return torch.mean(XX + YY - XY - YX)
