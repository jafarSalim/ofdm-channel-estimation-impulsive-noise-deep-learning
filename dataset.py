"""
Dataset Generation Module

Generates training, validation, and test datasets for DL-based
channel estimation. Data is generated at mixed SNR levels to ensure
the models generalize across operating conditions.

Each sample consists of:
    Input:  [Re(H_LS); Im(H_LS)] ∈ R^{2N}   (LS channel estimate)
    Target: [Re(H_true); Im(H_true)] ∈ R^{2N} (True channel response)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import Config
from ofdm_system import OFDMPipeline


class ChannelEstimationDataset(Dataset):
    """PyTorch Dataset for OFDM Channel Estimation.

    Stores pre-generated (LS_estimate, true_channel) pairs.
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def generate_dataset(cfg: Config, n_samples: int,
                     snr_range: tuple = (0, 30),
                     noise_type: str = 'impulsive',
                     desc: str = 'Generating') -> tuple:
    """Generate a dataset of (LS_estimate, true_channel) pairs.

    Training data is generated at uniformly random SNR values within
    the specified range, making the model SNR-agnostic.

    Args:
        cfg: Master configuration.
        n_samples: Number of samples to generate.
        snr_range: (min_snr_dB, max_snr_dB) — uniform sampling range.
        noise_type: 'impulsive' or 'gaussian'.
        desc: Progress bar description.

    Returns:
        inputs: LS estimates, shape (n_samples, 2N).
        targets: True channels, shape (n_samples, 2N).
    """
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)

    N = cfg.ofdm.n_subcarriers
    inputs = np.zeros((n_samples, 2 * N), dtype=np.float32)
    targets = np.zeros((n_samples, 2 * N), dtype=np.float32)

    for i in tqdm(range(n_samples), desc=desc, ncols=80):
        # Random SNR for each sample (mixed-SNR training)
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
        x_in, y_tgt = pipeline.generate_dataset_sample(snr_db, noise_type)
        inputs[i] = x_in
        targets[i] = y_tgt

    return inputs, targets


def create_dataloaders(cfg: Config,
                       noise_type: str = 'impulsive') -> dict:
    """Generate all datasets and create DataLoaders.

    Args:
        cfg: Master configuration.
        noise_type: Noise model for data generation.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders and raw arrays.
    """
    np.random.seed(cfg.training.seed)

    print("\n" + "=" * 60)
    print("  DATASET GENERATION")
    print("=" * 60)
    print(f"  Noise type:     {noise_type}")
    print(f"  Train samples:  {cfg.training.n_train:,}")
    print(f"  Val samples:    {cfg.training.n_val:,}")
    print(f"  Test samples:   {cfg.training.n_test:,}")
    print(f"  SNR range:      {cfg.training.train_snr_range} dB")
    print("=" * 60 + "\n")

    # Generate datasets
    train_in, train_tgt = generate_dataset(
        cfg, cfg.training.n_train,
        snr_range=cfg.training.train_snr_range,
        noise_type=noise_type, desc='Train data'
    )

    val_in, val_tgt = generate_dataset(
        cfg, cfg.training.n_val,
        snr_range=cfg.training.train_snr_range,
        noise_type=noise_type, desc='Val data'
    )

    test_in, test_tgt = generate_dataset(
        cfg, cfg.training.n_test,
        snr_range=cfg.training.train_snr_range,
        noise_type=noise_type, desc='Test data'
    )

    # Create PyTorch Datasets
    train_ds = ChannelEstimationDataset(train_in, train_tgt)
    val_ds = ChannelEstimationDataset(val_in, val_tgt)
    test_ds = ChannelEstimationDataset(test_in, test_tgt)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_data': (train_in, train_tgt),
        'val_data': (val_in, val_tgt),
        'test_data': (test_in, test_tgt),
    }
