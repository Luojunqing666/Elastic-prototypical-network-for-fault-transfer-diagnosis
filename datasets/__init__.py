"""
CWRU Bearing Dataset loader for meta-learning.

Supports loading vibration signal data from CSV files and creating
meta-learning task datasets via learn2learn.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset

from utils import l2_normalize, sample_label_shuffle


# ============================================================
# Data I/O
# ============================================================

def load_csv_signal(file_path, num_points=102400, header=0, shift_step=200):
    """
    Load vibration signal from a CSV file with overlapping reads if needed.

    Args:
        file_path: Path to the CSV file.
        num_points: Total number of data points to load.
        header: Starting row offset.
        shift_step: Row shift for overlapping reads when data is insufficient.

    Returns:
        1D numpy array of signal values.
    """
    data = pd.read_csv(file_path, header=header).values.reshape(-1)
    while data.shape[0] < num_points:
        header += shift_step
        data_extra = pd.read_csv(file_path, header=header).values.reshape(-1)
        data = np.concatenate((data, data_extra), axis=0)
    return data[:num_points]


def mat_to_csv(file_path, key_name='DE_time', output_path=None):
    """
    Convert a .mat file to .csv format.

    Args:
        file_path: Path to the .mat file.
        key_name: Key substring to search for in the .mat file.
        output_path: Output CSV path (default: same name with .csv extension).
    """
    mat_data = loadmat(file_path)
    keys = [k for k in mat_data.keys() if key_name in k]
    assert len(keys) == 1, f"Found {len(keys)} keys matching '{key_name}': {keys}"

    data = pd.DataFrame(mat_data[keys[0]])[0]
    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '.csv'
    data.to_csv(output_path, header=False, index=False)
    print(f"Converted: {file_path} -> {output_path}")


# ============================================================
# Dataset Configuration
# ============================================================

class CWRUConfig:
    """
    Configuration for CWRU dataset file paths.

    Users should modify `data_dir` and the task definitions (source/target)
    to match their local data organization.

    Args:
        data_dir: Root directory containing the CWRU CSV data files.
        source_files: List of CSV file paths for source domain (training).
        target_files: List of CSV file paths for target domain (testing).
    """

    def __init__(self, data_dir, source_files, target_files):
        self.data_dir = data_dir
        self.source_files = source_files
        self.target_files = target_files

    @property
    def num_source_classes(self):
        return len(self.source_files)

    @property
    def num_target_classes(self):
        return len(self.target_files)


# ============================================================
# Core Dataset Class
# ============================================================

class CWRUDataLoader:
    """
    Load and segment CWRU vibration data for meta-learning.

    Args:
        source_files: List of CSV file paths for source domain.
        target_files: List of CSV file paths for target domain.
    """

    def __init__(self, source_files, target_files):
        self.source_files = source_files
        self.target_files = target_files

    def load_data(self, train_mode=True, n_each_class=20, sample_len=1024, normalize=True):
        """
        Load and segment vibration data.

        Args:
            train_mode: If True, load source domain; otherwise target domain.
            n_each_class: Number of samples per class.
            sample_len: Length of each signal segment.
            normalize: Whether to apply L2 normalization.

        Returns:
            Tuple of (data, labels):
                data: [n_classes, n_each_class, sample_len]
                labels: [n_classes, n_each_class]
        """
        file_list = self.source_files if train_mode else self.target_files
        data_size = n_each_class * sample_len
        n_classes = len(file_list)

        data_set = []
        for file_path in file_list:
            signal = load_csv_signal(file_path, num_points=data_size, shift_step=200)
            segments = signal.reshape(-1, sample_len)
            if normalize:
                segments = l2_normalize(segments)
            data_set.append(segments)

        data_set = np.stack(data_set, axis=0).astype(np.float32)  # [n_classes, n_each, sample_len]
        labels = np.arange(n_classes, dtype=np.int32).reshape(n_classes, 1)
        labels = np.repeat(labels, n_each_class, axis=1)  # [n_classes, n_each]
        return data_set, labels


# ============================================================
# PyTorch Dataset for Meta-Learning
# ============================================================

N_TRAIN_EACH_CLASS = 20


class MetaLearningDataset(Dataset):
    """
    PyTorch Dataset for meta-learning with learn2learn.

    Args:
        config: CWRUConfig instance with file paths.
        mode: One of 'train', 'validation', 'test'.
        sample_len: Length of each signal segment.
    """

    def __init__(self, config, mode='train', sample_len=1024):
        super().__init__()
        self.sample_len = sample_len
        loader = CWRUDataLoader(config.source_files, config.target_files)
        self._load_data(loader, mode)

    def _load_data(self, loader, mode):
        if mode == 'train':
            data, labels = loader.load_data(
                train_mode=True, n_each_class=N_TRAIN_EACH_CLASS,
                sample_len=self.sample_len, normalize=True
            )
        else:
            data, labels = loader.load_data(
                train_mode=False, n_each_class=200,
                sample_len=self.sample_len, normalize=True
            )
            if mode == 'validation':
                data, labels = data[:, :100], labels[:, :100]
            elif mode == 'test':
                pass  # Use all data
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'train', 'validation', or 'test'.")

        # Reshape for meta-learning: [n_classes * n_each, 1, sample_len]
        self.x = data.reshape(-1, 1, self.sample_len)
        self.y = labels.reshape(-1)
        self.x, self.y = sample_label_shuffle(self.x, self.y)
        print(f"[{mode}] x: {self.x.shape}, y: {self.y.shape}")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
