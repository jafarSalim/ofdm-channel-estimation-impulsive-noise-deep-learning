<div align="center">

# Deep Learning Channel Estimation for OFDM Under Impulsive Noise

### PyTorch estimators tested against classic LS and MMSE under Middleton Class-A noise

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white)
![OFDM](https://img.shields.io/badge/System-OFDM%2016--QAM-0b2545)
![Figures](https://img.shields.io/badge/Figures-22-2ea44f)

![Overview](figures/overview.png)

</div>

## Contents

- [Key results](#key-results)
- [Flowchart](#flowchart)
- [Setup](#setup)
- [Result gallery](#result-gallery)
- [Run it](#run-it)
- [Files](#files)
- [Limits](#limits)

## Key results

![MSE vs SNR](figures/fig12_mse_vs_snr.png)

Impulsive noise, 16-QAM, 200 evaluation trials per SNR.

| Method | Parameters | NMSE at 20 dB | NMSE at 30 dB | BER at 20 dB |
|---|---:|---:|---:|---:|
| LS | none | -10.2 dB | -10.4 dB | 0.1126 |
| MMSE | none | -10.4 dB | -10.4 dB | 0.1112 |
| ResDNN | 6,285,696 | -20.0 dB | -23.5 dB | 0.0334 |
| ChannelCNN | 916,226 | **-20.5 dB** | **-24.7 dB** | **0.0328** |

What this shows:

- ChannelCNN beats ResDNN with about **7 times fewer parameters**.
- Both learned models keep improving at high SNR, while LS and MMSE stop near -10 dB.
- Results under Gaussian noise are very close to these, so this run does not show an advantage that is specific to impulsive noise.

## Flowchart

<p align="center"><img src="figures/flowchart.png" width="900" alt="Flowchart"></p>

## Setup

64 subcarriers, cyclic prefix 16, comb pilots every 4 subcarriers, 16-QAM, 10-tap Rayleigh channel, Class-A noise with A = 0.1 and Γ = 0.01, training SNR from 0 to 30 dB. Models learn a residual correction: `H_est = H_LS + f(H_LS)`.

## Result gallery

### Performance

| BER vs SNR | Gaussian vs impulsive noise |
|---|---|
| ![BER](figures/fig13_ber_vs_snr.png) | ![Noise compare](figures/fig24_gaussian_vs_impulsive.png) |

| Estimated channel | Constellations |
|---|---|
| ![Channel](figures/fig14_channel_estimation_comparison.png) | ![Constellations](figures/fig07_constellation_diagrams.png) |

### Signal model

| Noise distribution | Noise in time |
|---|---|
| ![PDF](figures/fig02_noise_pdf_comparison.png) | ![Time](figures/fig03_noise_time_domain.png) |

![System](figures/fig01_system_block_diagram.png)

### Models

| DNN | CNN |
|---|---|
| ![DNN](figures/fig08_dnn_architecture.png) | ![CNN](figures/fig09_cnn_architecture.png) |

### Error analysis

| Error per subcarrier | Error CDF |
|---|---|
| ![Subcarrier](figures/fig15_error_per_subcarrier.png) | ![CDF](figures/fig16_error_cdf.png) |

| NMSE heatmap | Error spread |
|---|---|
| ![Heatmap](figures/fig22_nmse_heatmap.png) | ![Box](figures/fig23_error_boxplot.png) |

| Effect of A | Effect of Gamma |
|---|---|
| ![A](figures/fig18_impact_of_A.png) | ![Gamma](figures/fig19_impact_of_Gamma.png) |

| Complexity | Summary |
|---|---|
| ![Complexity](figures/fig20_complexity_comparison.png) | ![Summary](figures/fig25_summary_table.png) |

## Run it

```bash
pip install -r requirements.txt

# The published run, about 40 minutes on a 2-core CPU
python main.py --quick --train_samples 30000 --epochs 25 --models ResDNN ChannelCNN

# Full settings in config.py: 100,000 samples, all four models
python main.py
```

The code also includes Transformer and Hybrid CNN-Attention models. They were not part of the published run.

## Files

| File | Purpose |
|---|---|
| `config.py` | All system and training settings |
| `ofdm_system.py` | OFDM link, channel, Class-A noise, LS and MMSE |
| `dataset.py` | Training data generation |
| `dl_models.py` | Four neural estimators |
| `trainer.py` | Training and evaluation |
| `visualize.py` | All figures |
| `main.py` | Full pipeline |

## Limits

- This is a simulation with synthetic channels, not over-the-air data.
- The published run used 30,000 training samples and 25 epochs, on the smaller side for these models.
- The -10 dB floor of LS and MMSE depends on how this code interpolates between pilots, so the gap to the neural models is not a general result.

## Author

**Jaafar Salim**, University of Technology, Iraq
