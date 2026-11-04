"""
Visualization Module — Publication-Quality Figures

Generates 20+ figures suitable for IEEE / Elsevier journal submissions.
All figures use consistent styling, LaTeX-compatible labels, and are
saved as both PNG (300 dpi) and PDF.

Figure List:
  1.  OFDM System Block Diagram
  2.  Middleton Class-A PDF vs Gaussian PDF
  3.  Time-domain Noise Realizations (Gaussian vs Impulsive)
  4.  Channel Power Delay Profile
  5.  Channel Frequency Response (Magnitude & Phase)
  6.  Pilot Pattern on OFDM Time-Frequency Grid
  7.  Constellation Diagrams (Tx, Rx-Gaussian, Rx-Impulsive)
  8.  DNN Architecture Diagram
  9.  CNN Architecture Diagram
  10. Training Loss Curves (all models)
  11. Learning Rate Schedule
  12. MSE vs SNR (all methods) — Main Result
  13. BER vs SNR (all methods) — Main Result
  14. Channel Estimation Comparison (Frequency Domain)
  15. Channel Estimation Error per Subcarrier
  16. CDF of Estimation Error
  17. Estimation Error Distribution (Histogram)
  18. Impact of Impulsive Parameter A on NMSE
  19. Impact of Γ on NMSE
  20. Computational Complexity Comparison (Bar Chart)
  21. Radar/Spider Chart — Multi-Metric Comparison
  22. Heatmap of NMSE across SNR and Methods
  23. Box Plot of Estimation Errors
  24. NMSE Comparison: Gaussian vs Impulsive Noise
  25. Performance Summary Table (as figure)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from config import Config
from ofdm_system import MiddletonClassA, OFDMPipeline

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Publication-quality style
STYLE = {
    'figure.figsize': (8, 5),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# Color palette — distinguishable, colorblind-friendly
COLORS = {
    'LS': '#e74c3c',         # Red
    'MMSE': '#3498db',       # Blue
    'ResDNN': '#2ecc71',     # Green
    'ChannelCNN': '#9b59b6', # Purple
    'Transformer': '#e67e22', # Orange
    'Hybrid': '#1abc9c',     # Teal
}

MARKERS = {
    'LS': 's',
    'MMSE': 'o',
    'ResDNN': '^',
    'ChannelCNN': 'D',
    'Transformer': 'v',
    'Hybrid': 'P',
}

LINE_STYLES = {
    'LS': '--',
    'MMSE': '-.',
    'ResDNN': '-',
    'ChannelCNN': '-',
    'Transformer': '-',
    'Hybrid': '-',
}


def _setup():
    plt.rcParams.update(STYLE)


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f'{name}.png'))
    fig.savefig(os.path.join(fig_dir, f'{name}.pdf'))
    plt.close(fig)
    print(f"    Saved: {name}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: OFDM System Block Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_system_block_diagram(fig_dir: str):
    """Draw OFDM system block diagram with DL estimator."""
    _setup()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Deep Learning-Based OFDM Channel Estimation System',
                 fontsize=15, fontweight='bold', pad=15)

    blocks = [
        (0.5, 3.0, 'QAM\nModulator'),
        (2.5, 3.0, 'Pilot\nInsertion'),
        (4.5, 3.0, 'IFFT +\nCP'),
        (6.5, 3.0, 'Channel\nh[n]'),
        (8.5, 3.0, 'CP Remove\n+ FFT'),
        (10.5, 3.0, 'Channel\nEstimation'),
        (12.5, 3.0, 'QAM\nDemod'),
    ]

    for (x, y, label) in blocks:
        color = '#3498db' if 'Estimation' not in label else '#e74c3c'
        box = FancyBboxPatch(
            (x - 0.7, y - 0.5), 1.4, 1.0,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Arrows between blocks
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + 0.7
        x2 = blocks[i + 1][0] - 0.7
        y = 3.0
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Noise block
    box_noise = FancyBboxPatch(
        (7.3, 1.0), 1.4, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8
    )
    ax.add_patch(box_noise)
    ax.text(8.0, 1.4, 'Middleton\nClass-A', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    ax.annotate('', xy=(8.0, 2.5), xytext=(8.0, 1.8),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(8.3, 2.15, '+', fontsize=14, fontweight='bold', color='red')

    # DL Estimator detail
    dl_blocks = [
        (9.8, 0.8, 'LS\nEstimate'),
        (11.3, 0.8, 'Deep\nNeural Net'),
        (12.8, 0.8, 'Refined\nH[k]'),
    ]
    for (x, y, label) in dl_blocks:
        box = FancyBboxPatch(
            (x - 0.55, y - 0.35), 1.1, 0.7,
            boxstyle="round,pad=0.05",
            facecolor='#2ecc71', edgecolor='black', linewidth=1, alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    for i in range(len(dl_blocks) - 1):
        x1 = dl_blocks[i][0] + 0.55
        x2 = dl_blocks[i + 1][0] - 0.55
        ax.annotate('', xy=(x2, 0.8), xytext=(x1, 0.8),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

    # Connection from main chain to DL detail
    ax.annotate('', xy=(10.5, 2.5), xytext=(10.5, 1.5),
                 arrowprops=dict(arrowstyle='<->', color='gray',
                                lw=1, ls='dashed'))

    _save(fig, fig_dir, 'fig01_system_block_diagram')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Middleton Class-A PDF vs Gaussian
# ═══════════════════════════════════════════════════════════════════════════════

def plot_noise_pdf(cfg: Config, fig_dir: str):
    """Compare PDFs of Gaussian and Middleton Class-A noise."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(-6, 6, 1000)
    noise_power = 1.0

    # Gaussian PDF
    gauss_pdf = (1 / np.sqrt(2 * np.pi * noise_power)) * \
                np.exp(-x ** 2 / (2 * noise_power))

    # Middleton Class-A PDFs for different A values
    A_values = [0.01, 0.1, 0.5, 1.0]
    colors_a = ['#e74c3c', '#e67e22', '#9b59b6', '#3498db']

    ax.semilogy(x, gauss_pdf, 'k-', linewidth=2.5, label='Gaussian (AWGN)')

    for A_val, color in zip(A_values, colors_a):
        from config import NoiseConfig
        noise_cfg = NoiseConfig(impulsive_A=A_val, impulsive_Gamma=cfg.noise.impulsive_Gamma)
        mca = MiddletonClassA(noise_cfg)
        pdf_vals = mca.pdf(x, noise_power)
        ax.semilogy(x, pdf_vals, color=color, linewidth=1.8,
                     label=f'Middleton (A={A_val})')

    ax.set_xlabel('Noise Amplitude')
    ax.set_ylabel('Probability Density (log scale)')
    ax.set_title('Middleton Class-A vs Gaussian Noise PDF')
    ax.legend(loc='upper right')
    ax.set_ylim([1e-6, 10])
    ax.set_xlim([-6, 6])
    ax.grid(True, alpha=0.3)

    _save(fig, fig_dir, 'fig02_noise_pdf_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Time-Domain Noise Realizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_noise_realizations(cfg: Config, fig_dir: str):
    """Show time-domain noise samples: Gaussian vs Impulsive."""
    _setup()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    n_samples = 500
    noise_power = 0.1

    np.random.seed(42)

    # Gaussian noise
    gauss_noise = MiddletonClassA.generate_gaussian(n_samples, noise_power)
    axes[0].stem(np.arange(n_samples), np.abs(gauss_noise),
                 linefmt='b-', markerfmt='bo', basefmt='k-',
                 label='Gaussian (AWGN)')
    axes[0].set_ylabel('|n(t)|')
    axes[0].set_title('Gaussian Noise (AWGN)', fontweight='bold')
    axes[0].set_ylim([0, 2.5])
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middleton Class-A noise
    mca = MiddletonClassA(cfg.noise)
    imp_noise = mca.generate(n_samples, noise_power)
    markerline, stemlines, baseline = axes[1].stem(
        np.arange(n_samples), np.abs(imp_noise),
        linefmt='r-', markerfmt='ro', basefmt='k-',
        label=f'Middleton Class-A (A={cfg.noise.impulsive_A})'
    )
    plt.setp(stemlines, linewidth=0.8)
    plt.setp(markerline, markersize=3)
    axes[1].set_ylabel('|n(t)|')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_title(
        f'Middleton Class-A Impulsive Noise (A={cfg.noise.impulsive_A}, '
        f'$\\Gamma$={cfg.noise.impulsive_Gamma})', fontweight='bold'
    )
    axes[1].set_ylim([0, 2.5])
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir, 'fig03_noise_time_domain')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Channel Power Delay Profile
# ═══════════════════════════════════════════════════════════════════════════════

def plot_power_delay_profile(cfg: Config, fig_dir: str):
    """Plot exponential power delay profile."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 5))

    L = cfg.channel.n_taps
    tau = cfg.channel.rms_delay_spread
    taps = np.arange(L)
    pdp = np.exp(-taps / tau)
    pdp /= np.sum(pdp)

    ax.stem(taps, 10 * np.log10(pdp + 1e-30),
            linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('Tap Index (l)')
    ax.set_ylabel('Normalized Power (dB)')
    ax.set_title(f'Exponential Power Delay Profile '
                 f'($\\tau_{{rms}}$ = {tau} samples, L = {L} taps)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    _save(fig, fig_dir, 'fig04_power_delay_profile')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Channel Frequency Response
# ═══════════════════════════════════════════════════════════════════════════════

def plot_channel_frequency_response(cfg: Config, fig_dir: str):
    """Plot magnitude and phase of channel frequency response."""
    _setup()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    np.random.seed(10)

    subcarrier_idx = np.arange(cfg.ofdm.n_subcarriers)

    # Plot 3 channel realizations
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i, color in enumerate(colors):
        h, H = pipeline.channel.generate()
        axes[0].plot(subcarrier_idx, 20 * np.log10(np.abs(H) + 1e-10),
                     color=color, linewidth=1.5, alpha=0.8,
                     label=f'Realization {i + 1}')
        axes[1].plot(subcarrier_idx, np.angle(H) * 180 / np.pi,
                     color=color, linewidth=1.5, alpha=0.8,
                     label=f'Realization {i + 1}')

    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Channel Frequency Response — Rayleigh Fading',
                       fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Subcarrier Index (k)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir, 'fig05_channel_frequency_response')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Pilot Pattern on OFDM Grid
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pilot_pattern(cfg: Config, fig_dir: str):
    """Visualize comb-type pilot insertion pattern on time-freq grid."""
    _setup()
    fig, ax = plt.subplots(figsize=(10, 5))

    N = cfg.ofdm.n_subcarriers
    n_symbols = 8  # Show multiple OFDM symbols

    # Data subcarriers
    for sym in range(n_symbols):
        for sc in range(N):
            if sc in cfg.ofdm.pilot_indices:
                ax.scatter(sym, sc, c='red', marker='d', s=30, zorder=3)
            else:
                ax.scatter(sym, sc, c='#3498db', marker='s', s=10,
                           alpha=0.4, zorder=2)

    ax.set_xlabel('OFDM Symbol Index')
    ax.set_ylabel('Subcarrier Index')
    ax.set_title('Comb-Type Pilot Pattern (OFDM Time-Frequency Grid)',
                 fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='red',
                   markersize=10, label='Pilot Subcarrier'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db',
                   markersize=10, label='Data Subcarrier', alpha=0.6),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.2)

    _save(fig, fig_dir, 'fig06_pilot_pattern')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Constellation Diagrams
# ═══════════════════════════════════════════════════════════════════════════════

def plot_constellation_diagrams(cfg: Config, fig_dir: str):
    """Show Tx, Rx (Gaussian), and Rx (Impulsive) constellations."""
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)

    # Tx constellation
    tx_symbols = pipeline.ofdm.qam.modulate(500)
    axes[0].scatter(tx_symbols.real, tx_symbols.imag,
                    c='#2ecc71', s=15, alpha=0.6)
    axes[0].set_title('Transmitted (16-QAM)', fontweight='bold')
    axes[0].set_xlabel('In-Phase')
    axes[0].set_ylabel('Quadrature')
    axes[0].set_xlim([-2, 2])
    axes[0].set_ylim([-2, 2])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Rx with Gaussian noise
    rx_gauss = []
    for _ in range(50):
        result = pipeline.transmit_receive(15.0, noise_type='gaussian')
        data_idx = cfg.ofdm.data_indices
        rx_gauss.extend(result['X_eq_ls'][data_idx])
    rx_gauss = np.array(rx_gauss)
    axes[1].scatter(rx_gauss.real, rx_gauss.imag,
                    c='#3498db', s=5, alpha=0.3)
    axes[1].set_title('Received (Gaussian Noise, 15 dB)', fontweight='bold')
    axes[1].set_xlabel('In-Phase')
    axes[1].set_xlim([-2, 2])
    axes[1].set_ylim([-2, 2])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    # Rx with Impulsive noise
    rx_imp = []
    for _ in range(50):
        result = pipeline.transmit_receive(15.0, noise_type='impulsive')
        data_idx = cfg.ofdm.data_indices
        rx_imp.extend(result['X_eq_ls'][data_idx])
    rx_imp = np.array(rx_imp)
    axes[2].scatter(rx_imp.real, rx_imp.imag,
                    c='#e74c3c', s=5, alpha=0.3)
    axes[2].set_title(
        f'Received (Impulsive, A={cfg.noise.impulsive_A}, 15 dB)',
        fontweight='bold'
    )
    axes[2].set_xlabel('In-Phase')
    axes[2].set_xlim([-2, 2])
    axes[2].set_ylim([-2, 2])
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')

    fig.tight_layout()
    _save(fig, fig_dir, 'fig07_constellation_diagrams')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: DNN Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dnn_architecture(cfg: Config, fig_dir: str):
    """Draw the ResDNN architecture diagram."""
    _setup()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('ResDNN Architecture for Channel Estimation',
                 fontsize=14, fontweight='bold', pad=15)

    # Layers
    layers = [
        (1.0, 2.5, f'Input\n{cfg.model.input_dim}', '#3498db'),
    ]
    x_pos = 2.5
    for i, h in enumerate(cfg.model.dnn_hidden):
        layers.append((x_pos, 2.5, f'FC+BN\n+ReLU\n{h}', '#2ecc71'))
        x_pos += 1.5
    layers.append((x_pos, 2.5, f'Output\n{cfg.model.output_dim}', '#e74c3c'))

    for (x, y, label, color) in layers:
        box = FancyBboxPatch(
            (x - 0.55, y - 0.6), 1.1, 1.2,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.85
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    # Arrows
    for i in range(len(layers) - 1):
        x1 = layers[i][0] + 0.55
        x2 = layers[i + 1][0] - 0.55
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Global residual connection
    x_start = layers[0][0]
    x_end = layers[-1][0]
    ax.annotate('',
                xy=(x_end, 1.5), xytext=(x_start, 1.5),
                arrowprops=dict(arrowstyle='->', color='#e67e22',
                               lw=2, ls='dashed',
                               connectionstyle='arc3,rad=-0.2'))
    ax.text((x_start + x_end) / 2, 0.9, 'Global Residual Connection (+)',
            ha='center', fontsize=10, color='#e67e22', fontweight='bold')

    _save(fig, fig_dir, 'fig08_dnn_architecture')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: CNN Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cnn_architecture(cfg: Config, fig_dir: str):
    """Draw the 1D-CNN architecture diagram."""
    _setup()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('1D-CNN Architecture with Dilated Convolutions',
                 fontsize=14, fontweight='bold', pad=15)

    blocks = [
        (1.0, 2.5, f'Input\n(B,2,{cfg.ofdm.n_subcarriers})', '#3498db'),
    ]

    x_pos = 3.0
    for i, (ch, ks) in enumerate(
        zip(cfg.model.cnn_channels, cfg.model.cnn_kernel_sizes)
    ):
        dilation = 2 ** min(i, 3)
        blocks.append(
            (x_pos, 2.5,
             f'Conv1D\n{ch}×{ks}\nd={dilation}',
             '#2ecc71')
        )
        x_pos += 1.8

    blocks.append((x_pos, 2.5, f'Conv 1×1\nOutput', '#e74c3c'))

    for (x, y, label, color) in blocks:
        box = FancyBboxPatch(
            (x - 0.7, y - 0.6), 1.4, 1.2,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.85
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + 0.7
        x2 = blocks[i + 1][0] - 0.7
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Global residual
    ax.annotate('',
                xy=(blocks[-1][0], 1.5), xytext=(blocks[0][0], 1.5),
                arrowprops=dict(arrowstyle='->', color='#e67e22',
                               lw=2, ls='dashed',
                               connectionstyle='arc3,rad=-0.15'))
    ax.text((blocks[0][0] + blocks[-1][0]) / 2, 0.85,
            'Global Residual Connection (+)',
            ha='center', fontsize=10, color='#e67e22', fontweight='bold')

    _save(fig, fig_dir, 'fig09_cnn_architecture')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Training Loss Curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(histories: dict, fig_dir: str):
    """Plot training and validation loss for all models."""
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        color = COLORS.get(name, '#333333')
        epochs = range(1, len(hist['train_loss']) + 1)

        axes[0].semilogy(epochs, hist['train_loss'],
                         color=color, linewidth=1.5, label=name)
        axes[1].semilogy(epochs, hist['val_loss'],
                         color=color, linewidth=1.5, label=name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss (MSE, log scale)')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss (MSE, log scale)')
    axes[1].set_title('Validation Loss', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir, 'fig10_training_curves')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lr_schedule(histories: dict, fig_dir: str):
    """Plot learning rate over epochs for all models."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, hist in histories.items():
        color = COLORS.get(name, '#333333')
        epochs = range(1, len(hist['lr']) + 1)
        ax.semilogy(epochs, hist['lr'], color=color, linewidth=1.5, label=name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.set_title('Learning Rate Schedule (ReduceLROnPlateau)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save(fig, fig_dir, 'fig11_learning_rate')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: MSE vs SNR (Main Result)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mse_vs_snr(eval_results: dict, fig_dir: str, noise_label: str = 'Impulsive'):
    """Plot NMSE vs SNR for all methods — primary performance figure."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    snr = eval_results['snr_range']
    mse = eval_results['mse']

    for name in eval_results['method_names']:
        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')
        ls = LINE_STYLES.get(name, '-')
        ax.semilogy(snr, mse[name], color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=8,
                    label=name, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Normalized MSE (log scale)')
    ax.set_title(f'Channel Estimation NMSE vs SNR ({noise_label} Noise)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(snr)

    _save(fig, fig_dir, 'fig12_mse_vs_snr')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: BER vs SNR (Main Result)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ber_vs_snr(eval_results: dict, fig_dir: str, noise_label: str = 'Impulsive'):
    """Plot BER vs SNR for all methods."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    snr = eval_results['snr_range']
    ber = eval_results['ber']

    for name in eval_results['method_names']:
        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')
        ls = LINE_STYLES.get(name, '-')
        ber_vals = np.clip(ber[name], 1e-6, 1)  # Clip for log scale
        ax.semilogy(snr, ber_vals, color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=8,
                    label=name, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (log scale)')
    ax.set_title(f'BER vs SNR ({noise_label} Noise, 16-QAM)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(snr)

    _save(fig, fig_dir, 'fig13_ber_vs_snr')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Channel Estimation Comparison (Frequency Domain)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_channel_estimation_comparison(cfg, models, device, fig_dir,
                                       snr_db=15.0):
    """Compare estimated vs true channel in frequency domain."""
    import torch
    _setup()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    np.random.seed(99)
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    result = pipeline.transmit_receive(snr_db, 'impulsive')

    H_true = result['H_true']
    H_ls = result['H_ls']
    H_mmse = result['H_mmse']
    N = cfg.ofdm.n_subcarriers
    subcarriers = np.arange(N)

    # DL estimates
    ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
    ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

    dl_estimates = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            pred = model(ls_tensor).cpu().numpy().flatten()
            dl_estimates[name] = pred[:N] + 1j * pred[N:]

    # Magnitude
    axes[0].plot(subcarriers, 20 * np.log10(np.abs(H_true) + 1e-10),
                 'k-', linewidth=2.5, label='True Channel')
    axes[0].plot(subcarriers, 20 * np.log10(np.abs(H_ls) + 1e-10),
                 color=COLORS['LS'], linestyle='--', linewidth=1.5, label='LS')
    axes[0].plot(subcarriers, 20 * np.log10(np.abs(H_mmse) + 1e-10),
                 color=COLORS['MMSE'], linestyle='-.', linewidth=1.5, label='MMSE')
    for name, H_est in dl_estimates.items():
        axes[0].plot(subcarriers, 20 * np.log10(np.abs(H_est) + 1e-10),
                     color=COLORS.get(name, '#333'),
                     linewidth=1.5, label=name)

    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'Channel Estimation Comparison (SNR = {snr_db} dB, '
                       f'Impulsive Noise)', fontweight='bold')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(subcarriers, np.angle(H_true) * 180 / np.pi,
                 'k-', linewidth=2.5, label='True Channel')
    axes[1].plot(subcarriers, np.angle(H_ls) * 180 / np.pi,
                 color=COLORS['LS'], linestyle='--', linewidth=1.5, label='LS')
    axes[1].plot(subcarriers, np.angle(H_mmse) * 180 / np.pi,
                 color=COLORS['MMSE'], linestyle='-.', linewidth=1.5, label='MMSE')
    for name, H_est in dl_estimates.items():
        axes[1].plot(subcarriers, np.angle(H_est) * 180 / np.pi,
                     color=COLORS.get(name, '#333'),
                     linewidth=1.5, label=name)

    axes[1].set_xlabel('Subcarrier Index (k)')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].legend(loc='lower left', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir, 'fig14_channel_estimation_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Estimation Error per Subcarrier
# ═══════════════════════════════════════════════════════════════════════════════

def plot_estimation_error_per_subcarrier(cfg, models, device, fig_dir,
                                          snr_db=15.0, n_trials=200):
    """Plot average estimation error magnitude per subcarrier."""
    import torch
    _setup()
    fig, ax = plt.subplots(figsize=(10, 5))

    N = cfg.ofdm.n_subcarriers
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    method_names = ['LS', 'MMSE'] + list(models.keys())
    error_accum = {name: np.zeros(N) for name in method_names}

    for model in models.values():
        model.eval()

    np.random.seed(42)
    for _ in range(n_trials):
        result = pipeline.transmit_receive(snr_db, 'impulsive')
        H_true = result['H_true']
        H_ls = result['H_ls']
        H_mmse = result['H_mmse']

        error_accum['LS'] += np.abs(H_true - H_ls) ** 2
        error_accum['MMSE'] += np.abs(H_true - H_mmse) ** 2

        ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
        ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

        with torch.no_grad():
            for name, model in models.items():
                pred = model(ls_tensor).cpu().numpy().flatten()
                H_dl = pred[:N] + 1j * pred[N:]
                error_accum[name] += np.abs(H_true - H_dl) ** 2

    subcarriers = np.arange(N)
    for name in method_names:
        color = COLORS.get(name, '#333')
        ls = LINE_STYLES.get(name, '-')
        ax.plot(subcarriers, 10 * np.log10(error_accum[name] / n_trials + 1e-15),
                color=color, linestyle=ls, linewidth=1.5, label=name)

    # Mark pilot positions
    pilot_idx = cfg.ofdm.pilot_indices
    ax.scatter(pilot_idx, np.full_like(pilot_idx, ax.get_ylim()[0], dtype=float),
               marker='|', c='red', s=100, zorder=5, label='Pilot positions')

    ax.set_xlabel('Subcarrier Index (k)')
    ax.set_ylabel('MSE per Subcarrier (dB)')
    ax.set_title(f'Estimation Error Distribution Across Subcarriers '
                 f'(SNR = {snr_db} dB)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    _save(fig, fig_dir, 'fig15_error_per_subcarrier')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 16: CDF of Estimation Error
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_cdf(cfg, models, device, fig_dir,
                   snr_db=15.0, n_trials=500):
    """Plot Cumulative Distribution Function of estimation error."""
    import torch
    _setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    N = cfg.ofdm.n_subcarriers
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    method_names = ['LS', 'MMSE'] + list(models.keys())
    all_errors = {name: [] for name in method_names}

    for model in models.values():
        model.eval()

    np.random.seed(42)
    for _ in range(n_trials):
        result = pipeline.transmit_receive(snr_db, 'impulsive')
        H_true = result['H_true']
        H_ls = result['H_ls']
        H_mmse = result['H_mmse']

        nmse_ls = np.sum(np.abs(H_true - H_ls) ** 2) / np.sum(np.abs(H_true) ** 2)
        nmse_mmse = np.sum(np.abs(H_true - H_mmse) ** 2) / np.sum(np.abs(H_true) ** 2)
        all_errors['LS'].append(nmse_ls)
        all_errors['MMSE'].append(nmse_mmse)

        ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
        ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

        with torch.no_grad():
            for name, model in models.items():
                pred = model(ls_tensor).cpu().numpy().flatten()
                H_dl = pred[:N] + 1j * pred[N:]
                nmse = np.sum(np.abs(H_true - H_dl) ** 2) / np.sum(np.abs(H_true) ** 2)
                all_errors[name].append(nmse)

    for name in method_names:
        color = COLORS.get(name, '#333')
        ls = LINE_STYLES.get(name, '-')
        sorted_err = np.sort(all_errors[name])
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(10 * np.log10(sorted_err + 1e-15), cdf,
                color=color, linestyle=ls, linewidth=2, label=name)

    ax.set_xlabel('NMSE (dB)')
    ax.set_ylabel('CDF')
    ax.set_title(f'CDF of Channel Estimation NMSE (SNR = {snr_db} dB)',
                 fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    _save(fig, fig_dir, 'fig16_error_cdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 17: Estimation Error Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_histogram(cfg, models, device, fig_dir,
                          snr_db=15.0, n_trials=500):
    """Plot histogram of estimation errors for all methods."""
    import torch
    _setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    N = cfg.ofdm.n_subcarriers
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    method_names = ['LS', 'MMSE'] + list(models.keys())
    all_errors = {name: [] for name in method_names}

    for model in models.values():
        model.eval()

    np.random.seed(42)
    for _ in range(n_trials):
        result = pipeline.transmit_receive(snr_db, 'impulsive')
        H_true = result['H_true']
        H_ls = result['H_ls']
        H_mmse = result['H_mmse']

        all_errors['LS'].append(
            np.sum(np.abs(H_true - H_ls) ** 2) / np.sum(np.abs(H_true) ** 2))
        all_errors['MMSE'].append(
            np.sum(np.abs(H_true - H_mmse) ** 2) / np.sum(np.abs(H_true) ** 2))

        ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
        ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

        with torch.no_grad():
            for name, model in models.items():
                pred = model(ls_tensor).cpu().numpy().flatten()
                H_dl = pred[:N] + 1j * pred[N:]
                all_errors[name].append(
                    np.sum(np.abs(H_true - H_dl) ** 2) / np.sum(np.abs(H_true) ** 2))

    for name in method_names:
        color = COLORS.get(name, '#333')
        ax.hist(10 * np.log10(np.array(all_errors[name]) + 1e-15),
                bins=40, alpha=0.4, color=color, label=name,
                edgecolor='black', linewidth=0.5)

    ax.set_xlabel('NMSE (dB)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Estimation Errors (SNR = {snr_db} dB)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save(fig, fig_dir, 'fig17_error_histogram')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 18: Impact of Impulsive Parameter A
# ═══════════════════════════════════════════════════════════════════════════════

def plot_impact_of_A(impact_A_results: dict, fig_dir: str):
    """Plot NMSE vs impulsive parameter A."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    A_values = impact_A_results['A_values']
    mse = impact_A_results['mse']

    for name in mse.keys():
        color = COLORS.get(name, '#333')
        marker = MARKERS.get(name, 'o')
        ls = LINE_STYLES.get(name, '-')
        ax.semilogy(A_values, mse[name], color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=8, label=name,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Impulsive Index (A)')
    ax.set_ylabel('NMSE (log scale)')
    ax.set_title(
        f'Impact of Impulsive Index A on NMSE '
        f'(SNR = {impact_A_results["snr_db"]} dB)',
        fontweight='bold'
    )
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xscale('log')

    _save(fig, fig_dir, 'fig18_impact_of_A')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 19: Impact of Γ
# ═══════════════════════════════════════════════════════════════════════════════

def plot_impact_of_Gamma(impact_G_results: dict, fig_dir: str):
    """Plot NMSE vs Gaussian-to-impulsive ratio Γ."""
    _setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    G_values = impact_G_results['Gamma_values']
    mse = impact_G_results['mse']

    for name in mse.keys():
        color = COLORS.get(name, '#333')
        marker = MARKERS.get(name, 'o')
        ls = LINE_STYLES.get(name, '-')
        ax.semilogy(G_values, mse[name], color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=8, label=name,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('$\\Gamma$ (Gaussian-to-Impulsive Ratio)')
    ax.set_ylabel('NMSE (log scale)')
    ax.set_title(
        f'Impact of $\\Gamma$ on NMSE (SNR = {impact_G_results["snr_db"]} dB)',
        fontweight='bold'
    )
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xscale('log')

    _save(fig, fig_dir, 'fig19_impact_of_Gamma')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 20: Computational Complexity Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_complexity_comparison(models: dict, fig_dir: str):
    """Bar chart comparing model sizes and computational complexity."""
    from dl_models import count_parameters
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = list(models.keys())
    params = [count_parameters(m) / 1000 for m in models.values()]  # in thousands

    # Parameter count
    colors = [COLORS.get(n, '#333') for n in names]
    bars1 = axes[0].bar(names, params, color=colors, edgecolor='black',
                        linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel('Parameters (×1000)')
    axes[0].set_title('Model Size Comparison', fontweight='bold')
    for bar, val in zip(bars1, params):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}K', ha='center', va='bottom', fontsize=10)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Inference time estimate (forward pass)
    import torch
    import time
    times = []
    dummy = torch.randn(100, list(models.values())[0].output_layer.in_features
                         if hasattr(list(models.values())[0], 'output_layer')
                         else 128)
    for name, model in models.items():
        model.eval()
        model.cpu()
        inp_size = 128  # 2 * N_subcarriers
        dummy_in = torch.randn(100, inp_size)
        # Warm up
        with torch.no_grad():
            _ = model(dummy_in)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_in)
        t1 = time.time()
        times.append((t1 - t0) / 50 * 1000)  # ms per 100 samples

    bars2 = axes[1].bar(names, times, color=colors, edgecolor='black',
                        linewidth=1.2, alpha=0.85)
    axes[1].set_ylabel('Inference Time (ms / 100 samples)')
    axes[1].set_title('Inference Speed Comparison', fontweight='bold')
    for bar, val in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir, 'fig20_complexity_comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 21: Radar / Spider Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(eval_results: dict, models: dict, fig_dir: str):
    """Multi-metric radar chart comparing all methods."""
    from dl_models import count_parameters
    _setup()

    method_names = eval_results['method_names']
    mse_data = eval_results['mse']
    ber_data = eval_results['ber']

    # Metrics: lower is better, normalize to [0, 1] where 1 = best
    metrics = ['Low SNR\nNMSE', 'High SNR\nNMSE', 'Low SNR\nBER',
               'High SNR\nBER', 'Compactness']

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name in method_names:
        # Compute scores (inverse of error, normalized)
        low_snr_mse = 1 / (mse_data[name][0] + 1e-10)
        high_snr_mse = 1 / (mse_data[name][-1] + 1e-10)
        low_snr_ber = 1 / (ber_data[name][0] + 1e-10)
        high_snr_ber = 1 / (ber_data[name][-1] + 1e-10)
        if name in models:
            compactness = 1 / (count_parameters(models[name]) + 1)
        else:
            compactness = 0.5  # Classical methods

        values = [low_snr_mse, high_snr_mse, low_snr_ber,
                  high_snr_ber, compactness]

        # Normalize
        max_val = max(values) if max(values) > 0 else 1
        values = [v / max_val for v in values]
        values += values[:1]

        color = COLORS.get(name, '#333')
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title('Multi-Metric Performance Comparison',
                 fontweight='bold', pad=30, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    _save(fig, fig_dir, 'fig21_radar_chart')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 22: NMSE Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_nmse_heatmap(eval_results: dict, fig_dir: str):
    """Heatmap of NMSE values across methods and SNR levels."""
    _setup()
    fig, ax = plt.subplots(figsize=(10, 5))

    snr = eval_results['snr_range']
    method_names = eval_results['method_names']
    mse = eval_results['mse']

    # Build matrix
    matrix = np.array([10 * np.log10(mse[name] + 1e-15) for name in method_names])

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                   interpolation='nearest')
    ax.set_xticks(range(len(snr)))
    ax.set_xticklabels([f'{s}' for s in snr])
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_xlabel('SNR (dB)')
    ax.set_title('NMSE Heatmap (dB) — Lower is Better', fontweight='bold')

    # Annotate cells
    for i in range(len(method_names)):
        for j in range(len(snr)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if val > np.median(matrix) else 'black')

    fig.colorbar(im, ax=ax, label='NMSE (dB)')
    fig.tight_layout()
    _save(fig, fig_dir, 'fig22_nmse_heatmap')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 23: Box Plot of Estimation Errors
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_boxplot(cfg, models, device, fig_dir,
                       snr_db=15.0, n_trials=300):
    """Box plot of estimation errors across methods."""
    import torch
    _setup()
    fig, ax = plt.subplots(figsize=(10, 6))

    N = cfg.ofdm.n_subcarriers
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    method_names = ['LS', 'MMSE'] + list(models.keys())
    all_errors = {name: [] for name in method_names}

    for model in models.values():
        model.eval()

    np.random.seed(42)
    for _ in range(n_trials):
        result = pipeline.transmit_receive(snr_db, 'impulsive')
        H_true = result['H_true']
        H_ls = result['H_ls']
        H_mmse = result['H_mmse']

        all_errors['LS'].append(
            10 * np.log10(np.sum(np.abs(H_true - H_ls) ** 2) /
                          np.sum(np.abs(H_true) ** 2) + 1e-15))
        all_errors['MMSE'].append(
            10 * np.log10(np.sum(np.abs(H_true - H_mmse) ** 2) /
                          np.sum(np.abs(H_true) ** 2) + 1e-15))

        ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
        ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

        with torch.no_grad():
            for name, model in models.items():
                pred = model(ls_tensor).cpu().numpy().flatten()
                H_dl = pred[:N] + 1j * pred[N:]
                all_errors[name].append(
                    10 * np.log10(np.sum(np.abs(H_true - H_dl) ** 2) /
                                  np.sum(np.abs(H_true) ** 2) + 1e-15))

    box_data = [all_errors[name] for name in method_names]
    colors_list = [COLORS.get(name, '#333') for name in method_names]

    bp = ax.boxplot(box_data, tick_labels=method_names, patch_artist=True,
                    widths=0.5, showfliers=True, notch=True)

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('NMSE (dB)')
    ax.set_title(f'Distribution of Channel Estimation Errors '
                 f'(SNR = {snr_db} dB)', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    _save(fig, fig_dir, 'fig23_error_boxplot')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 24: Gaussian vs Impulsive Noise Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gaussian_vs_impulsive(eval_gauss: dict, eval_imp: dict, fig_dir: str):
    """Compare NMSE under Gaussian vs Impulsive noise for all methods."""
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    snr = eval_gauss['snr_range']

    # Gaussian
    for name in eval_gauss['method_names']:
        color = COLORS.get(name, '#333')
        marker = MARKERS.get(name, 'o')
        ls = LINE_STYLES.get(name, '-')
        axes[0].semilogy(snr, eval_gauss['mse'][name], color=color,
                         marker=marker, linestyle=ls, linewidth=2,
                         markersize=8, label=name)

    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('NMSE (log scale)')
    axes[0].set_title('NMSE under Gaussian Noise (AWGN)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].set_xticks(snr)

    # Impulsive
    for name in eval_imp['method_names']:
        color = COLORS.get(name, '#333')
        marker = MARKERS.get(name, 'o')
        ls = LINE_STYLES.get(name, '-')
        axes[1].semilogy(snr, eval_imp['mse'][name], color=color,
                         marker=marker, linestyle=ls, linewidth=2,
                         markersize=8, label=name)

    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('NMSE (log scale)')
    axes[1].set_title(f'NMSE under Impulsive Noise (A={0.1})', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].set_xticks(snr)

    fig.tight_layout()
    _save(fig, fig_dir, 'fig24_gaussian_vs_impulsive')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 25: Performance Summary Table
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary_table(eval_results: dict, models: dict, fig_dir: str):
    """Create a publication-quality summary table as a figure."""
    from dl_models import count_parameters
    _setup()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    method_names = eval_results['method_names']
    snr = eval_results['snr_range']
    mse = eval_results['mse']
    ber = eval_results['ber']

    # Table data
    col_labels = ['Method', 'Parameters'] + \
                 [f'NMSE\n{s} dB' for s in snr] + \
                 [f'BER\n{snr[-1]} dB']

    table_data = []
    for name in method_names:
        if name in models:
            n_params = f'{count_parameters(models[name]):,}'
        else:
            n_params = 'N/A'

        row = [name, n_params]
        for s_idx in range(len(snr)):
            row.append(f'{mse[name][s_idx]:.4f}')
        row.append(f'{ber[name][-1]:.4f}')
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] % 2 == 0:
            cell.set_facecolor('#ecf0f1')

    ax.set_title('Performance Summary Table', fontweight='bold',
                 fontsize=14, pad=20)

    _save(fig, fig_dir, 'fig25_summary_table')


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION: Generate All Figures
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(cfg: Config, models: dict, device,
                         histories: dict,
                         eval_imp: dict, eval_gauss: dict,
                         impact_A: dict, impact_G: dict):
    """Generate all 25 publication-quality figures.

    Args:
        cfg: Configuration.
        models: Dict of trained DL models.
        device: Compute device.
        histories: Training histories per model.
        eval_imp: Evaluation results under impulsive noise.
        eval_gauss: Evaluation results under Gaussian noise.
        impact_A: Impact of A evaluation results.
        impact_G: Impact of Γ evaluation results.
    """
    fig_dir = cfg.fig_dir
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60 + "\n")

    # Signal Processing & System Figures
    print("  [Signal Processing Figures]")
    plot_system_block_diagram(fig_dir)
    plot_noise_pdf(cfg, fig_dir)
    plot_noise_realizations(cfg, fig_dir)
    plot_power_delay_profile(cfg, fig_dir)
    plot_channel_frequency_response(cfg, fig_dir)
    plot_pilot_pattern(cfg, fig_dir)
    plot_constellation_diagrams(cfg, fig_dir)

    # Architecture Diagrams
    print("\n  [Architecture Diagrams]")
    plot_dnn_architecture(cfg, fig_dir)
    plot_cnn_architecture(cfg, fig_dir)

    # Training Figures
    print("\n  [Training Figures]")
    plot_training_curves(histories, fig_dir)
    plot_lr_schedule(histories, fig_dir)

    # Main Performance Figures
    print("\n  [Performance Figures]")
    plot_mse_vs_snr(eval_imp, fig_dir, 'Impulsive')
    plot_ber_vs_snr(eval_imp, fig_dir, 'Impulsive')
    plot_channel_estimation_comparison(cfg, models, device, fig_dir)
    plot_estimation_error_per_subcarrier(cfg, models, device, fig_dir)

    # Statistical Analysis Figures
    print("\n  [Statistical Analysis Figures]")
    plot_error_cdf(cfg, models, device, fig_dir)
    plot_error_histogram(cfg, models, device, fig_dir)
    plot_impact_of_A(impact_A, fig_dir)
    plot_impact_of_Gamma(impact_G, fig_dir)

    # Comparison Figures
    print("\n  [Comparison Figures]")
    plot_complexity_comparison(models, fig_dir)
    plot_radar_chart(eval_imp, models, fig_dir)
    plot_nmse_heatmap(eval_imp, fig_dir)
    plot_error_boxplot(cfg, models, device, fig_dir)
    plot_gaussian_vs_impulsive(eval_gauss, eval_imp, fig_dir)
    plot_summary_table(eval_imp, models, fig_dir)

    print(f"\n  All figures saved to: {os.path.abspath(fig_dir)}/")
    print("=" * 60)
