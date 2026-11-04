"""
Training Pipeline Module

Implements:
  - Model training with validation monitoring
  - Early stopping based on validation loss
  - Learning rate scheduling (ReduceLROnPlateau)
  - Model checkpointing (best model saving)
  - Training history logging
  - Per-SNR evaluation for MSE and BER curves
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from ofdm_system import OFDMPipeline, MiddletonClassA
from dl_models import create_model, count_parameters


def get_device(cfg: Config) -> torch.device:
    """Resolve compute device."""
    if cfg.training.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg.training.device)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model: nn.Module, train_loader, val_loader,
                cfg: Config, device: torch.device) -> dict:
    """Train a DL model for channel estimation.

    Args:
        model: PyTorch model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Master configuration.
        device: Compute device.

    Returns:
        history: Dictionary with training/validation loss per epoch.
    """
    model = model.to(device)
    model_name = getattr(model, 'name', model.__class__.__name__)

    print(f"\n{'-' * 60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")
    print(f"{'-' * 60}")

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg.training.lr_factor,
        patience=cfg.training.lr_patience,
        min_lr=cfg.training.min_lr,
    )
    criterion = nn.MSELoss()

    # Checkpointing
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f'{model_name}_best.pth')

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
    }

    # Early stopping
    patience_counter = 0

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()

        # ── Training Phase ──
        model.train()
        train_losses = []
        for batch_in, batch_tgt in train_loader:
            batch_in = batch_in.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()
            pred = model(batch_in)
            loss = criterion(pred, batch_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ── Validation Phase ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_in, batch_tgt in val_loader:
                batch_in = batch_in.to(device)
                batch_tgt = batch_tgt.to(device)
                pred = model(batch_in)
                loss = criterion(pred, batch_tgt)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Log
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Checkpointing (save best model)
        if avg_val_loss < history['best_val_loss']:
            history['best_val_loss'] = avg_val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Progress reporting
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{cfg.training.epochs} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.1e} | "
                f"Best: {history['best_val_loss']:.6f} (ep {history['best_epoch']}) | "
                f"{elapsed:.1f}s"
            )

        # Early stopping
        if patience_counter >= cfg.training.patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.training.patience} epochs)")
            break

    # Load best model
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  Loaded best model from epoch {history['best_epoch']} "
          f"(val_loss = {history['best_val_loss']:.6f})")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# PER-SNR EVALUATION (MSE and BER)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_per_snr(models: dict, cfg: Config, device: torch.device,
                     n_trials: int = 2000,
                     noise_type: str = 'impulsive') -> dict:
    """Evaluate all methods (LS, MMSE, DL models) at each SNR point.

    Computes:
        - MSE:  E[ ||H_true - H_est||² ] / E[ ||H_true||² ]  (NMSE)
        - BER:  Bit Error Rate after ZF equalization

    Args:
        models: Dict of {name: nn.Module} for DL models.
        cfg: Master configuration.
        device: Compute device.
        n_trials: Monte Carlo trials per SNR point.
        noise_type: 'impulsive' or 'gaussian'.

    Returns:
        results: Dict with MSE and BER arrays for each method.
    """
    pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, cfg.noise)
    snr_range = cfg.snr_eval
    N = cfg.ofdm.n_subcarriers

    method_names = ['LS', 'MMSE'] + list(models.keys())
    mse_results = {name: np.zeros(len(snr_range)) for name in method_names}
    ber_results = {name: np.zeros(len(snr_range)) for name in method_names}

    # Set all DL models to eval mode
    for model in models.values():
        model.eval()

    print("\n" + "=" * 60)
    print(f"  EVALUATION ({noise_type} noise, {n_trials} trials/SNR)")
    print("=" * 60)

    for s, snr_db in enumerate(snr_range):
        mse_accum = {name: 0.0 for name in method_names}
        ber_accum = {name: 0.0 for name in method_names}

        for trial in tqdm(range(n_trials),
                          desc=f'SNR={snr_db:2d} dB', ncols=80, leave=False):
            result = pipeline.transmit_receive(snr_db, noise_type)
            H_true = result['H_true']
            H_ls = result['H_ls']
            H_mmse = result['H_mmse']
            Y_freq = result['Y_freq']
            X_freq = result['X_freq']

            # True channel power for normalization
            h_power = np.sum(np.abs(H_true) ** 2)

            # ── LS MSE ──
            mse_accum['LS'] += np.sum(np.abs(H_true - H_ls) ** 2) / h_power

            # ── MMSE MSE ──
            mse_accum['MMSE'] += np.sum(np.abs(H_true - H_mmse) ** 2) / h_power

            # ── DL Models MSE ──
            ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
            ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

            with torch.no_grad():
                for name, model in models.items():
                    pred = model(ls_tensor).cpu().numpy().flatten()
                    H_dl = pred[:N] + 1j * pred[N:]
                    mse_accum[name] += np.sum(
                        np.abs(H_true - H_dl) ** 2
                    ) / h_power

            # ── BER Computation ──
            data_idx = cfg.ofdm.data_indices
            tx_bits = pipeline.ofdm.qam.symbols_to_bits(
                pipeline.ofdm.qam.demodulate(X_freq[data_idx])
            )

            # BER for LS
            X_eq = Y_freq[data_idx] / (H_ls[data_idx] + 1e-10)
            rx_bits = pipeline.ofdm.qam.symbols_to_bits(
                pipeline.ofdm.qam.demodulate(X_eq)
            )
            ber_accum['LS'] += np.mean(tx_bits != rx_bits)

            # BER for MMSE
            X_eq = Y_freq[data_idx] / (H_mmse[data_idx] + 1e-10)
            rx_bits = pipeline.ofdm.qam.symbols_to_bits(
                pipeline.ofdm.qam.demodulate(X_eq)
            )
            ber_accum['MMSE'] += np.mean(tx_bits != rx_bits)

            # BER for DL models
            with torch.no_grad():
                for name, model in models.items():
                    pred = model(ls_tensor).cpu().numpy().flatten()
                    H_dl = pred[:N] + 1j * pred[N:]
                    X_eq = Y_freq[data_idx] / (H_dl[data_idx] + 1e-10)
                    rx_bits = pipeline.ofdm.qam.symbols_to_bits(
                        pipeline.ofdm.qam.demodulate(X_eq)
                    )
                    ber_accum[name] += np.mean(tx_bits != rx_bits)

        # Average over trials
        for name in method_names:
            mse_results[name][s] = mse_accum[name] / n_trials
            ber_results[name][s] = ber_accum[name] / n_trials

        # Print results for this SNR
        mse_str = " | ".join(
            f"{name}: {mse_results[name][s]:.4f}" for name in method_names
        )
        print(f"  SNR={snr_db:2d} dB | NMSE: {mse_str}")

    return {
        'snr_range': snr_range,
        'mse': mse_results,
        'ber': ber_results,
        'method_names': method_names,
    }


def evaluate_impact_of_A(models: dict, cfg: Config, device: torch.device,
                         snr_db: float = 15.0,
                         A_values: np.ndarray = None,
                         n_trials: int = 1000) -> dict:
    """Evaluate impact of impulsive noise parameter A on estimation.

    Args:
        models: Dict of DL models.
        cfg: Configuration.
        device: Compute device.
        snr_db: Fixed SNR for this evaluation.
        A_values: Array of impulsive index values to test.
        n_trials: Trials per A value.

    Returns:
        Dictionary with MSE vs A for each method.
    """
    from config import NoiseConfig

    if A_values is None:
        A_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0])

    N = cfg.ofdm.n_subcarriers
    method_names = ['LS', 'MMSE'] + list(models.keys())
    mse_results = {name: np.zeros(len(A_values)) for name in method_names}

    for model in models.values():
        model.eval()

    print(f"\n  Evaluating impact of A at SNR = {snr_db} dB")

    for a_idx, A_val in enumerate(A_values):
        noise_cfg = NoiseConfig(impulsive_A=A_val, impulsive_Gamma=cfg.noise.impulsive_Gamma)
        pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, noise_cfg)

        mse_accum = {name: 0.0 for name in method_names}

        for _ in tqdm(range(n_trials), desc=f'A={A_val:.2f}',
                      ncols=80, leave=False):
            result = pipeline.transmit_receive(snr_db, 'impulsive')
            H_true = result['H_true']
            H_ls = result['H_ls']
            H_mmse = result['H_mmse']
            h_power = np.sum(np.abs(H_true) ** 2)

            mse_accum['LS'] += np.sum(np.abs(H_true - H_ls) ** 2) / h_power
            mse_accum['MMSE'] += np.sum(np.abs(H_true - H_mmse) ** 2) / h_power

            ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
            ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

            with torch.no_grad():
                for name, model in models.items():
                    pred = model(ls_tensor).cpu().numpy().flatten()
                    H_dl = pred[:N] + 1j * pred[N:]
                    mse_accum[name] += np.sum(
                        np.abs(H_true - H_dl) ** 2
                    ) / h_power

        for name in method_names:
            mse_results[name][a_idx] = mse_accum[name] / n_trials

    return {'A_values': A_values, 'mse': mse_results, 'snr_db': snr_db}


def evaluate_impact_of_Gamma(models: dict, cfg: Config, device: torch.device,
                             snr_db: float = 15.0,
                             Gamma_values: np.ndarray = None,
                             n_trials: int = 1000) -> dict:
    """Evaluate impact of Γ (Gaussian-to-impulsive ratio) on estimation."""
    from config import NoiseConfig

    if Gamma_values is None:
        Gamma_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5])

    N = cfg.ofdm.n_subcarriers
    method_names = ['LS', 'MMSE'] + list(models.keys())
    mse_results = {name: np.zeros(len(Gamma_values)) for name in method_names}

    for model in models.values():
        model.eval()

    print(f"\n  Evaluating impact of Gamma at SNR = {snr_db} dB")

    for g_idx, G_val in enumerate(Gamma_values):
        noise_cfg = NoiseConfig(impulsive_A=cfg.noise.impulsive_A, impulsive_Gamma=G_val)
        pipeline = OFDMPipeline(cfg.ofdm, cfg.channel, noise_cfg)

        mse_accum = {name: 0.0 for name in method_names}

        for _ in tqdm(range(n_trials), desc=f'Gamma={G_val:.3f}',
                      ncols=80, leave=False):
            result = pipeline.transmit_receive(snr_db, 'impulsive')
            H_true = result['H_true']
            H_ls = result['H_ls']
            H_mmse = result['H_mmse']
            h_power = np.sum(np.abs(H_true) ** 2)

            mse_accum['LS'] += np.sum(np.abs(H_true - H_ls) ** 2) / h_power
            mse_accum['MMSE'] += np.sum(np.abs(H_true - H_mmse) ** 2) / h_power

            ls_input = np.concatenate([H_ls.real, H_ls.imag]).astype(np.float32)
            ls_tensor = torch.FloatTensor(ls_input).unsqueeze(0).to(device)

            with torch.no_grad():
                for name, model in models.items():
                    pred = model(ls_tensor).cpu().numpy().flatten()
                    H_dl = pred[:N] + 1j * pred[N:]
                    mse_accum[name] += np.sum(
                        np.abs(H_true - H_dl) ** 2
                    ) / h_power

        for name in method_names:
            mse_results[name][g_idx] = mse_accum[name] / n_trials

    return {'Gamma_values': Gamma_values, 'mse': mse_results, 'snr_db': snr_db}
