"""
Configuration Module for Deep Learning-Based Channel Estimation
for OFDM Systems Under Impulsive Noise

All system, channel, noise, training, and model hyperparameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class OFDMConfig:
    """OFDM System Parameters"""
    n_subcarriers: int = 64             # FFT size (N)
    n_cp: int = 16                       # Cyclic prefix length
    pilot_spacing: int = 4               # Comb-type pilot spacing
    modulation: str = '16QAM'            # '4QAM', '16QAM', '64QAM'

    @property
    def n_pilots(self) -> int:
        return self.n_subcarriers // self.pilot_spacing

    @property
    def n_data(self) -> int:
        return self.n_subcarriers - self.n_pilots

    @property
    def pilot_indices(self) -> np.ndarray:
        return np.arange(0, self.n_subcarriers, self.pilot_spacing)

    @property
    def data_indices(self) -> np.ndarray:
        return np.setdiff1d(
            np.arange(self.n_subcarriers), self.pilot_indices
        )


@dataclass
class ChannelConfig:
    """Multipath Fading Channel Parameters"""
    n_taps: int = 10                     # Number of channel taps (L)
    rms_delay_spread: float = 3.0        # RMS delay spread in samples
    channel_type: str = 'rayleigh'       # 'rayleigh' or 'rician'
    rician_k_factor: float = 3.0         # Rician K-factor (dB)
    normalize: bool = True               # Normalize channel power to 1


@dataclass
class NoiseConfig:
    """Middleton Class-A Impulsive Noise Parameters"""
    impulsive_A: float = 0.1             # Impulsive index (overlap factor)
    impulsive_Gamma: float = 0.01        # Gaussian-to-impulsive power ratio
    n_terms: int = 15                    # Truncation terms for PDF summation


@dataclass
class TrainingConfig:
    """Training Pipeline Parameters"""
    n_train: int = 100000                # Training samples
    n_val: int = 20000                   # Validation samples
    n_test: int = 20000                  # Test samples
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 25                   # Early stopping patience
    lr_patience: int = 10                # LR scheduler patience
    lr_factor: float = 0.5              # LR reduction factor
    min_lr: float = 1e-6
    train_snr_range: Tuple[int, int] = (0, 30)  # SNR range for training
    seed: int = 42
    device: str = 'auto'                 # 'cpu', 'cuda', 'auto'


@dataclass
class ModelConfig:
    """Deep Learning Model Architectures"""
    # Input/Output dimensions = 2 * n_subcarriers (real + imaginary)
    input_dim: int = 128
    output_dim: int = 128

    # ── DNN (Fully Connected with Residual Connections) ──
    dnn_hidden: List[int] = field(
        default_factory=lambda: [512, 1024, 1024, 512, 256]
    )
    dnn_dropout: float = 0.1

    # ── CNN (1D Convolutional Network) ──
    cnn_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256, 128, 64]
    )
    cnn_kernel_sizes: List[int] = field(
        default_factory=lambda: [7, 5, 5, 3, 3]
    )
    cnn_dropout: float = 0.1

    # ── Transformer / Self-Attention ──
    attn_embed_dim: int = 128
    attn_num_heads: int = 4
    attn_num_layers: int = 3
    attn_ff_dim: int = 256
    attn_dropout: float = 0.1


@dataclass
class Config:
    """Master Configuration"""
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # SNR range for performance evaluation
    snr_eval: np.ndarray = field(
        default_factory=lambda: np.arange(0, 35, 5)
    )

    # Figures output directory
    fig_dir: str = 'figures'

    # Results output directory
    results_dir: str = 'results'

    # Model checkpoints directory
    checkpoint_dir: str = 'checkpoints'

    def __post_init__(self):
        """Synchronize input/output dimensions with OFDM config."""
        self.model.input_dim = 2 * self.ofdm.n_subcarriers
        self.model.output_dim = 2 * self.ofdm.n_subcarriers
