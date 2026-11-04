"""
Deep Learning Models for OFDM Channel Estimation

Implements three advanced architectures:
  1. ResDNN    — Fully connected network with residual skip connections
  2. ChannelCNN — 1D convolutional network with dilated convolutions
  3. AttentionEstimator — Multi-head self-attention (Transformer-style)

All models follow the residual learning paradigm:
    H_est = H_LS + f_θ(H_LS)

where f_θ learns the estimation error rather than the channel directly.
This approach significantly accelerates convergence and improves accuracy.

Input:  [Re(H_LS); Im(H_LS)] ∈ R^{2N}   (LS estimate, real-valued)
Output: [Re(H);    Im(H)]    ∈ R^{2N}   (Refined channel estimate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Fully connected residual block with BatchNorm and dropout.

         x ──→ Linear → BN → LeakyReLU → Dropout → Linear → BN ──(+)──→ out
         │                                                          │
         └────────────────── identity / project ───────────────────┘
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Identity()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = self.activation(out + identity)
        return out


class ConvBlock(nn.Module):
    """1D Convolutional block with BatchNorm, activation, and residual."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.activation(out + identity)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: ResDNN — Residual Deep Neural Network
# ═══════════════════════════════════════════════════════════════════════════════

class ResDNN(nn.Module):
    """Residual DNN for Channel Estimation.

    Architecture:
        Input (2N) → [ResBlock × K] → Output (2N)

    Uses global residual learning:
        output = input + network(input)

    This forces the network to learn the LS estimation error.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.name = 'ResDNN'

        layers = []
        in_dim = cfg.input_dim
        for hidden_dim in cfg.dnn_hidden:
            layers.append(ResidualBlock(in_dim, hidden_dim, cfg.dnn_dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, cfg.output_dim)

    def forward(self, x):
        """
        Args:
            x: LS channel estimate [Re(H_LS); Im(H_LS)], shape (B, 2N)

        Returns:
            Refined estimate [Re(H); Im(H)], shape (B, 2N)
        """
        residual = x  # Global skip connection
        out = self.backbone(x)
        out = self.output_layer(out)
        return out + residual  # Residual learning


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: ChannelCNN — 1D Convolutional Network
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelCNN(nn.Module):
    """1D CNN for Channel Estimation with Dilated Convolutions.

    Architecture:
        Input (B, 2N) → Reshape (B, 2, N) → [ConvBlock × K] → Conv1x1 → (B, 2N)

    Uses two input channels (real and imaginary parts) and
    dilated convolutions to capture long-range frequency correlations.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.name = 'ChannelCNN'
        self.N = cfg.input_dim // 2  # Number of subcarriers

        # Build convolutional blocks with increasing dilation
        conv_layers = []
        in_ch = 2  # Real + Imaginary
        for i, (out_ch, ks) in enumerate(
            zip(cfg.cnn_channels, cfg.cnn_kernel_sizes)
        ):
            dilation = 2 ** min(i, 3)  # 1, 2, 4, 8, 8, ...
            conv_layers.append(
                ConvBlock(in_ch, out_ch, ks, dilation, cfg.cnn_dropout)
            )
            in_ch = out_ch

        self.conv_backbone = nn.Sequential(*conv_layers)

        # Final 1x1 convolution to map back to 2 channels
        self.output_conv = nn.Conv1d(in_ch, 2, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: LS estimate, shape (B, 2N)

        Returns:
            Refined estimate, shape (B, 2N)
        """
        B = x.size(0)
        residual = x

        # Reshape to (B, 2, N): channel dim = {real, imag}
        x_2d = x.view(B, 2, self.N)

        out = self.conv_backbone(x_2d)
        out = self.output_conv(out)

        # Reshape back to (B, 2N)
        out = out.view(B, -1)

        return out + residual  # Residual learning


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: AttentionEstimator — Transformer-Based
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for subcarrier positions."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm feed-forward
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionEstimator(nn.Module):
    """Transformer-based Channel Estimator.

    Treats each subcarrier as a token with 2 features (real, imag).
    Self-attention captures inter-subcarrier correlations in the
    frequency domain — naturally suited for frequency-selective channels.

    Architecture:
        Input (B, 2N) → Reshape (B, N, 2) → Linear proj → PE
        → [TransformerBlock × L] → Linear → Reshape (B, 2N)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.name = 'Transformer'
        self.N = cfg.input_dim // 2

        d_model = cfg.attn_embed_dim
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.N)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, cfg.attn_num_heads,
                cfg.attn_ff_dim, cfg.attn_dropout
            )
            for _ in range(cfg.attn_num_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x):
        """
        Args:
            x: LS estimate, shape (B, 2N)

        Returns:
            Refined estimate, shape (B, 2N)
        """
        B = x.size(0)
        residual = x

        # Reshape: (B, 2N) → (B, N, 2)
        x_seq = x.view(B, 2, self.N).permute(0, 2, 1)  # (B, N, 2)

        # Project to d_model dimensions
        x_proj = self.input_proj(x_seq)  # (B, N, d_model)
        x_proj = self.pos_enc(x_proj)

        # Transformer blocks
        for block in self.blocks:
            x_proj = block(x_proj)

        x_proj = self.output_norm(x_proj)

        # Project back to 2 features and reshape
        out = self.output_proj(x_proj)  # (B, N, 2)
        out = out.permute(0, 2, 1).contiguous().view(B, -1)  # (B, 2N)

        return out + residual  # Residual learning


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4: Hybrid CNN-Attention Model
# ═══════════════════════════════════════════════════════════════════════════════

class HybridCNNAttention(nn.Module):
    """Hybrid CNN + Self-Attention for Channel Estimation.

    Combines local feature extraction (CNN) with global context
    modeling (self-attention) for superior estimation performance.

    Architecture:
        Input → CNN feature extraction → Self-Attention → Output
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.name = 'Hybrid'
        self.N = cfg.input_dim // 2

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )

        # Self-attention on CNN features
        self.attn = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(64)

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 2, kernel_size=1),
        )

    def forward(self, x):
        B = x.size(0)
        residual = x

        # (B, 2N) → (B, 2, N)
        x_2d = x.view(B, 2, self.N)

        # CNN feature extraction: (B, 2, N) → (B, 64, N)
        cnn_feat = self.cnn(x_2d)

        # Self-attention: (B, 64, N) → (B, N, 64)
        attn_in = cnn_feat.permute(0, 2, 1)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        attn_out = self.attn_norm(attn_out + attn_in)

        # Output: (B, N, 64) → (B, 64, N) → (B, 2, N) → (B, 2N)
        out = attn_out.permute(0, 2, 1)
        out = self.output_conv(out)
        out = out.view(B, -1)

        return out + residual


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(model_type: str, cfg: ModelConfig) -> nn.Module:
    """Factory function to create models by name.

    Args:
        model_type: One of 'ResDNN', 'ChannelCNN', 'Transformer', 'Hybrid'.
        cfg: Model configuration.

    Returns:
        Instantiated PyTorch model.
    """
    models = {
        'ResDNN': ResDNN,
        'ChannelCNN': ChannelCNN,
        'Transformer': AttentionEstimator,
        'Hybrid': HybridCNNAttention,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type](cfg)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
