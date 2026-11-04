"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Deep Learning-Based Channel Estimation for OFDM Systems                   ║
║  Under Middleton Class-A Impulsive Noise                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  This code implements a complete research pipeline for DL-based channel    ║
║  estimation in OFDM systems operating under impulsive noise:               ║
║                                                                            ║
║  Models:                                                                   ║
║    1. Least Squares (LS) — baseline                                        ║
║    2. Linear MMSE (LMMSE) — classical optimal                              ║
║    3. ResDNN — Residual Deep Neural Network                                ║
║    4. ChannelCNN — 1D CNN with dilated convolutions                        ║
║    5. Transformer — Multi-head self-attention                              ║
║    6. Hybrid — CNN + Self-Attention                                        ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • 25 publication-quality figures (PNG + PDF)                             ║
║    • MSE & BER performance curves                                          ║
║    • Robustness analysis (varying A, Γ)                                    ║
║    • Trained model checkpoints                                             ║
║                                                                            ║
║  Usage:                                                                    ║
║    python main.py                                                          ║
║    python main.py --quick          # Quick demo (fewer samples)            ║
║    python main.py --epochs 300     # Custom training epochs                ║
║                                                                            ║
║  Target Journals:                                                          ║
║    IEEE Trans. Wireless Communications, IEEE Comm. Letters,                ║
║    Physical Communication (Elsevier)                                       ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import json

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import Config
from dataset import create_dataloaders
from dl_models import create_model, count_parameters
from trainer import (
    get_device, train_model,
    evaluate_per_snr, evaluate_impact_of_A, evaluate_impact_of_Gamma
)
from visualize import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description='DL-Based Channel Estimation for OFDM Under Impulsive Noise'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick demo mode (reduced samples & epochs)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs')
    parser.add_argument('--train_samples', type=int, default=None,
                        help='Override number of training samples')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Compute device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_eval_trials', type=int, default=None,
                        help='Monte Carlo trials per SNR for evaluation')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and load existing checkpoints')
    parser.add_argument('--models', nargs='+',
                        default=['ResDNN', 'ChannelCNN', 'Transformer', 'Hybrid'],
                        help='Models to train and evaluate')
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════════
    cfg = Config()

    if args.quick:
        # Quick demo settings for fast testing
        cfg.training.n_train = 10000
        cfg.training.n_val = 2000
        cfg.training.n_test = 2000
        cfg.training.epochs = 30
        cfg.training.batch_size = 128
        n_eval_trials = 200
        n_impact_trials = 100
    else:
        n_eval_trials = args.n_eval_trials or 2000
        n_impact_trials = 1000

    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.train_samples:
        cfg.training.n_train = args.train_samples
    if args.device:
        cfg.training.device = args.device
    cfg.training.seed = args.seed

    # Set random seeds
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    device = get_device(cfg)

    # Create output directories
    os.makedirs(cfg.fig_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("  DEEP LEARNING-BASED CHANNEL ESTIMATION FOR OFDM SYSTEMS")
    print("  UNDER MIDDLETON CLASS-A IMPULSIVE NOISE")
    print("=" * 70)
    print(f"\n  OFDM Subcarriers:     {cfg.ofdm.n_subcarriers}")
    print(f"  Cyclic Prefix:        {cfg.ofdm.n_cp}")
    print(f"  Pilot Spacing:        {cfg.ofdm.pilot_spacing} "
          f"({cfg.ofdm.n_pilots} pilots)")
    print(f"  Modulation:           {cfg.ofdm.modulation}")
    print(f"  Channel Taps:         {cfg.channel.n_taps}")
    print(f"  RMS Delay Spread:     {cfg.channel.rms_delay_spread} samples")
    print(f"  Impulsive Index (A):  {cfg.noise.impulsive_A}")
    print(f"  Gamma:                {cfg.noise.impulsive_Gamma}")
    print(f"  Device:               {device}")
    print(f"  Training Samples:     {cfg.training.n_train:,}")
    print(f"  Epochs:               {cfg.training.epochs}")
    print(f"  Models:               {args.models}")
    print(f"  Eval Trials/SNR:      {n_eval_trials}")
    print("=" * 70)

    # ======================================================================
    # STEP 1: GENERATE TRAINING DATA
    # ======================================================================
    print("\n>> STEP 1: Generating Training Data...")
    data = create_dataloaders(cfg, noise_type='impulsive')

    # ======================================================================
    # STEP 2: CREATE AND TRAIN MODELS
    # ======================================================================
    print("\n>> STEP 2: Training Deep Learning Models...")

    trained_models = {}
    histories = {}

    for model_name in args.models:
        model = create_model(model_name, cfg.model)
        print(f"\n  Model: {model_name} | "
              f"Parameters: {count_parameters(model):,}")

        if args.skip_training:
            ckpt_path = os.path.join(cfg.checkpoint_dir,
                                     f'{model_name}_best.pth')
            if os.path.exists(ckpt_path):
                model.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True)
                )
                print(f"  Loaded checkpoint: {ckpt_path}")
                trained_models[model_name] = model.to(device)
                histories[model_name] = {'train_loss': [], 'val_loss': [],
                                          'lr': []}
                continue
            else:
                print(f"  No checkpoint found, training from scratch...")

        history = train_model(
            model, data['train_loader'], data['val_loader'], cfg, device
        )
        trained_models[model_name] = model.to(device)
        histories[model_name] = history

    # ======================================================================
    # STEP 3: EVALUATE UNDER IMPULSIVE NOISE
    # ======================================================================
    print("\n>> STEP 3: Evaluating Under Impulsive Noise...")
    eval_impulsive = evaluate_per_snr(
        trained_models, cfg, device,
        n_trials=n_eval_trials, noise_type='impulsive'
    )

    # ======================================================================
    # STEP 4: EVALUATE UNDER GAUSSIAN NOISE (COMPARISON)
    # ======================================================================
    print("\n>> STEP 4: Evaluating Under Gaussian Noise (Comparison)...")
    eval_gaussian = evaluate_per_snr(
        trained_models, cfg, device,
        n_trials=n_eval_trials, noise_type='gaussian'
    )

    # ======================================================================
    # STEP 5: ROBUSTNESS ANALYSIS
    # ======================================================================
    print("\n>> STEP 5: Robustness Analysis...")

    impact_A = evaluate_impact_of_A(
        trained_models, cfg, device,
        snr_db=15.0, n_trials=n_impact_trials
    )

    impact_Gamma = evaluate_impact_of_Gamma(
        trained_models, cfg, device,
        snr_db=15.0, n_trials=n_impact_trials
    )

    # ======================================================================
    # STEP 6: GENERATE ALL FIGURES
    # ======================================================================
    print("\n>> STEP 6: Generating 25 Publication-Quality Figures...")

    generate_all_figures(
        cfg=cfg,
        models=trained_models,
        device=device,
        histories=histories,
        eval_imp=eval_impulsive,
        eval_gauss=eval_gaussian,
        impact_A=impact_A,
        impact_G=impact_Gamma,
    )

    # ======================================================================
    # STEP 7: SAVE NUMERICAL RESULTS
    # ======================================================================
    print("\n>> STEP 7: Saving Numerical Results...")

    results = {
        'snr_range': cfg.snr_eval.tolist(),
        'impulsive_mse': {k: v.tolist() for k, v in eval_impulsive['mse'].items()},
        'impulsive_ber': {k: v.tolist() for k, v in eval_impulsive['ber'].items()},
        'gaussian_mse': {k: v.tolist() for k, v in eval_gaussian['mse'].items()},
        'gaussian_ber': {k: v.tolist() for k, v in eval_gaussian['ber'].items()},
        'impact_A_values': impact_A['A_values'].tolist(),
        'impact_A_mse': {k: v.tolist() for k, v in impact_A['mse'].items()},
        'impact_Gamma_values': impact_Gamma['Gamma_values'].tolist(),
        'impact_Gamma_mse': {k: v.tolist() for k, v in impact_Gamma['mse'].items()},
        'model_parameters': {
            name: count_parameters(model)
            for name, model in trained_models.items()
        },
        'config': {
            'n_subcarriers': cfg.ofdm.n_subcarriers,
            'n_cp': cfg.ofdm.n_cp,
            'pilot_spacing': cfg.ofdm.pilot_spacing,
            'modulation': cfg.ofdm.modulation,
            'n_taps': cfg.channel.n_taps,
            'impulsive_A': cfg.noise.impulsive_A,
            'impulsive_Gamma': cfg.noise.impulsive_Gamma,
            'train_samples': cfg.training.n_train,
            'epochs': cfg.training.epochs,
        }
    }

    results_path = os.path.join(cfg.results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Figures:     {cfg.fig_dir}/")
    print(f"  Results:     {cfg.results_dir}/")
    print(f"  Checkpoints: {cfg.checkpoint_dir}/")

    print("\n  +--- NMSE at 15 dB (Impulsive) ---+")
    snr_15_idx = np.argmin(np.abs(cfg.snr_eval - 15))
    for name in eval_impulsive['method_names']:
        mse_val = eval_impulsive['mse'][name][snr_15_idx]
        ber_val = eval_impulsive['ber'][name][snr_15_idx]
        bar = "#" * int(max(0, min(50, -10 * np.log10(mse_val + 1e-15))))
        print(f"  | {name:15s} | NMSE: {mse_val:.6f} | "
              f"BER: {ber_val:.6f} | {bar}")
    print("  +-----------------------------------+")

    print("\n  Best DL model improvement over LS (NMSE at 15 dB):")
    ls_mse = eval_impulsive['mse']['LS'][snr_15_idx]
    for name in args.models:
        dl_mse = eval_impulsive['mse'][name][snr_15_idx]
        improvement = 10 * np.log10(ls_mse / (dl_mse + 1e-15))
        print(f"    {name}: {improvement:.2f} dB improvement")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
