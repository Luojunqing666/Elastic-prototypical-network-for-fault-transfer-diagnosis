"""Utility functions for EProtoNet."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from sklearn.preprocessing import normalize


def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 2023):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    print(f"Random seed set to: {seed}")


def weights_init(module):
    """
    Initialize network weights (Kaiming-style for Conv1d, uniform for BN).

    Args:
        module: A PyTorch module.
    """
    if isinstance(module, nn.Conv1d):
        n = module.kernel_size[0] * module.out_channels
        module.weight.data.normal_(mean=0, std=np.sqrt(2.0 / float(n)))
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.fill_(0)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.01)
        if module.bias is not None:
            module.bias.data.fill_(0)


def l2_normalize(x):
    """
    L2 normalize each sample in the array.

    Args:
        x: Array of shape (n_samples, length).

    Returns:
        L2-normalized array.
    """
    return np.asarray(normalize(x, norm='l2', axis=1))


def accuracy(predictions, targets):
    """
    Compute classification accuracy.

    Args:
        predictions: Model output logits [batch_size, num_classes].
        targets: Ground truth labels [batch_size].

    Returns:
        Accuracy as a float tensor.
    """
    preds = predictions.argmax(dim=-1).view(targets.shape)
    return (preds == targets).sum().float() / targets.shape[0]


def one_hot_embedding(labels, num_classes):
    """
    Convert integer labels to one-hot encoding.

    Args:
        labels: Integer tensor or array of class indices.
        num_classes: Total number of classes.

    Returns:
        One-hot encoded array.
    """
    res = np.zeros(np.shape(labels) + (num_classes,), dtype=np.float32)
    it = np.nditer(labels, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def sample_label_shuffle(data, label):
    """
    Shuffle data and labels together.

    Args:
        data: Array of shape [num_samples, ...].
        label: Array of shape [num_samples].

    Returns:
        Tuple of (shuffled_data, shuffled_label).
    """
    index = np.arange(len(data))
    np.random.shuffle(index)
    return data[index], label[index]
