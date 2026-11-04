"""
OFDM System Simulation Module

Implements:
  - QAM Modulation / Demodulation (4-QAM, 16-QAM, 64-QAM)
  - OFDM Modulator (IFFT + Cyclic Prefix) / Demodulator (CP removal + FFT)
  - Comb-type Pilot Insertion & Extraction
  - Multipath Rayleigh / Rician Fading Channel
  - Middleton Class-A Impulsive Noise Model
  - Least Squares (LS) Channel Estimator
  - Linear Minimum Mean Square Error (LMMSE) Channel Estimator
  - Linear / Spline Interpolation for pilot-to-data mapping

Mathematical References:
  [1] Middleton, D. "Non-Gaussian Noise Models in Signal Processing for
      Telecommunications", IEEE Trans. on Electromagnetic Compatibility, 1999.
  [2] Zhidkov, S.V. "Impulsive Noise Suppression in OFDM-Based Communication
      Systems", IEEE Trans. on Consumer Electronics, 2003.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import factorial
from config import OFDMConfig, ChannelConfig, NoiseConfig


# ═══════════════════════════════════════════════════════════════════════════════
# QAM MODULATION / DEMODULATION
# ═══════════════════════════════════════════════════════════════════════════════

class QAMModulator:
    """M-QAM Modulator and Demodulator.

    Supports 4-QAM (QPSK), 16-QAM, 64-QAM with Gray coding.
    Constellation is normalized to unit average power.
    """

    def __init__(self, modulation: str = '16QAM'):
        self.modulation = modulation
        self.M = int(modulation.replace('QAM', ''))
        self.m = int(np.sqrt(self.M))  # Points per dimension
        self.bits_per_symbol = int(np.log2(self.M))
        self.constellation = self._create_constellation()

    def _create_constellation(self) -> np.ndarray:
        """Create normalized Gray-coded QAM constellation."""
        # Generate constellation points: {-(m-1), -(m-3), ..., (m-3), (m-1)}
        levels = np.arange(-(self.m - 1), self.m, 2).astype(float)
        # Create 2D grid
        real_part, imag_part = np.meshgrid(levels, levels)
        constellation = (real_part + 1j * imag_part).flatten()
        # Normalize to unit average power: E[|s|^2] = 1
        avg_power = np.mean(np.abs(constellation) ** 2)
        constellation /= np.sqrt(avg_power)
        return constellation

    def modulate(self, n_symbols: int) -> np.ndarray:
        """Generate random QAM symbols.

        Args:
            n_symbols: Number of symbols to generate.

        Returns:
            Complex QAM symbols of shape (n_symbols,).
        """
        indices = np.random.randint(0, self.M, size=n_symbols)
        return self.constellation[indices]

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Hard-decision demodulation (minimum distance).

        Args:
            symbols: Received complex symbols.

        Returns:
            Indices of closest constellation points.
        """
        # Compute distance to all constellation points
        distances = np.abs(
            symbols[:, np.newaxis] - self.constellation[np.newaxis, :]
        )
        return np.argmin(distances, axis=1)

    def symbols_to_bits(self, indices: np.ndarray) -> np.ndarray:
        """Convert symbol indices to bit stream."""
        bits = []
        for idx in indices:
            b = format(idx, f'0{self.bits_per_symbol}b')
            bits.extend([int(x) for x in b])
        return np.array(bits)


# ═══════════════════════════════════════════════════════════════════════════════
# OFDM MODULATOR / DEMODULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class OFDMSystem:
    """OFDM Transceiver with Comb-Type Pilot Insertion.

    Signal model (frequency domain, per subcarrier k):
        Y[k] = H[k] · X[k] + W[k]

    where H[k] is the channel frequency response, X[k] is the transmitted
    symbol, and W[k] is additive noise (possibly impulsive).
    """

    def __init__(self, config: OFDMConfig):
        self.cfg = config
        self.qam = QAMModulator(config.modulation)

    def generate_tx_signal(self) -> tuple:
        """Generate one OFDM symbol with pilots.

        Returns:
            X_freq: Frequency-domain transmitted signal (N,) complex
            data_symbols: Data symbols at data subcarrier positions
            pilot_symbols: Pilot symbols at pilot positions
        """
        N = self.cfg.n_subcarriers
        X_freq = np.zeros(N, dtype=complex)

        # Insert pilot symbols (known BPSK pilots: +1)
        pilot_indices = self.cfg.pilot_indices
        pilot_symbols = np.ones(len(pilot_indices), dtype=complex)
        X_freq[pilot_indices] = pilot_symbols

        # Insert random QAM data symbols
        data_indices = self.cfg.data_indices
        data_symbols = self.qam.modulate(len(data_indices))
        X_freq[data_indices] = data_symbols

        return X_freq, data_symbols, pilot_symbols

    def ofdm_modulate(self, X_freq: np.ndarray) -> np.ndarray:
        """IFFT + Cyclic Prefix insertion.

        Args:
            X_freq: Frequency-domain symbols (N,)

        Returns:
            x_time: Time-domain OFDM signal with CP (N + N_cp,)
        """
        # IFFT to get time-domain signal
        x_time = np.fft.ifft(X_freq, n=self.cfg.n_subcarriers)
        # Add cyclic prefix
        x_cp = np.concatenate([x_time[-self.cfg.n_cp:], x_time])
        return x_cp

    def ofdm_demodulate(self, y_time: np.ndarray) -> np.ndarray:
        """Remove Cyclic Prefix + FFT.

        Args:
            y_time: Received time-domain signal with CP (N + N_cp,)

        Returns:
            Y_freq: Frequency-domain received signal (N,)
        """
        # Remove cyclic prefix
        y_no_cp = y_time[self.cfg.n_cp:]
        # FFT to get frequency-domain signal
        Y_freq = np.fft.fft(y_no_cp, n=self.cfg.n_subcarriers)
        return Y_freq

    def extract_pilots(self, Y_freq: np.ndarray) -> tuple:
        """Extract pilot observations and known pilot values.

        Returns:
            Y_pilots: Received signal at pilot positions
            X_pilots: Transmitted pilot symbols
        """
        pilot_idx = self.cfg.pilot_indices
        Y_pilots = Y_freq[pilot_idx]
        X_pilots = np.ones(len(pilot_idx), dtype=complex)  # Known pilots
        return Y_pilots, X_pilots


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPATH FADING CHANNEL
# ═══════════════════════════════════════════════════════════════════════════════

class MultipathChannel:
    """Multipath Rayleigh / Rician Fading Channel.

    Power Delay Profile (exponential):
        σ²_l = exp(-l / τ_rms) / Σ_l exp(-l / τ_rms),  l = 0, ..., L-1

    Channel taps:
        h[l] ~ CN(0, σ²_l)         (Rayleigh)
        h[0] ~ CN(μ, σ²_0)         (Rician, LOS component on first tap)

    Frequency-domain channel:
        H[k] = Σ_l h[l] · exp(-j2πkl/N),  k = 0, ..., N-1
    """

    def __init__(self, ofdm_cfg: OFDMConfig, ch_cfg: ChannelConfig):
        self.ofdm_cfg = ofdm_cfg
        self.ch_cfg = ch_cfg
        self.pdp = self._compute_pdp()

    def _compute_pdp(self) -> np.ndarray:
        """Compute normalized exponential Power Delay Profile."""
        L = self.ch_cfg.n_taps
        tau_rms = self.ch_cfg.rms_delay_spread
        taps = np.arange(L)
        pdp = np.exp(-taps / tau_rms)
        pdp /= np.sum(pdp)  # Normalize so total power = 1
        return pdp

    def generate(self) -> tuple:
        """Generate one channel realization.

        Returns:
            h: Time-domain channel impulse response (L,)
            H: Frequency-domain channel response (N,)
        """
        L = self.ch_cfg.n_taps
        N = self.ofdm_cfg.n_subcarriers

        if self.ch_cfg.channel_type == 'rayleigh':
            # h[l] ~ CN(0, σ²_l)
            h = (np.random.randn(L) + 1j * np.random.randn(L)) \
                * np.sqrt(self.pdp / 2)

        elif self.ch_cfg.channel_type == 'rician':
            # Rician: LOS component on first tap
            K = 10 ** (self.ch_cfg.rician_k_factor / 10)  # Linear
            # Diffuse component
            h = (np.random.randn(L) + 1j * np.random.randn(L)) \
                * np.sqrt(self.pdp / (2 * (1 + K)))
            # LOS component on first tap
            h[0] += np.sqrt(K / (1 + K) * self.pdp[0])
        else:
            raise ValueError(f"Unknown channel type: {self.ch_cfg.channel_type}")

        if self.ch_cfg.normalize:
            h /= np.sqrt(np.sum(np.abs(h) ** 2))

        # Frequency-domain response via N-point FFT
        H = np.fft.fft(h, n=N)

        return h, H

    def apply(self, x_time: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Apply channel to time-domain signal (linear convolution).

        Args:
            x_time: Transmitted time-domain signal (with CP)
            h: Channel impulse response

        Returns:
            y_time: Received signal (same length as x_time)
        """
        y_conv = np.convolve(x_time, h, mode='full')
        return y_conv[:len(x_time)]

    def get_correlation_matrix(self) -> np.ndarray:
        """Compute frequency-domain channel correlation matrix R_HH.

        R_HH = F_L · diag(σ²_0, ..., σ²_{L-1}) · F_L^H

        where F_L is the first L columns of the N×N DFT matrix.

        Returns:
            R_HH: Channel correlation matrix (N × N)
        """
        N = self.ofdm_cfg.n_subcarriers
        L = self.ch_cfg.n_taps

        # DFT matrix (first L columns)
        k = np.arange(N).reshape(-1, 1)
        l = np.arange(L).reshape(1, -1)
        F_L = np.exp(-1j * 2 * np.pi * k * l / N)

        # R_HH = F_L · diag(pdp) · F_L^H
        R_HH = F_L @ np.diag(self.pdp) @ F_L.conj().T

        return R_HH


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLETON CLASS-A IMPULSIVE NOISE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MiddletonClassA:
    """Middleton Class-A Impulsive Noise Generator.

    The noise is modeled as a mixture of Gaussian components:

        p(n) = Σ_{m=0}^{M} α_m · CN(0, σ²_m)

    where:
        α_m = exp(-A) · A^m / m!               (Poisson weights)
        σ²_m = σ² · (m/A + Γ) / (1 + Γ)        (Component variance)
        A:    Impulsive index (small A → more impulsive)
        Γ:    Gaussian-to-impulsive power ratio (small Γ → more impulsive)
        σ²:   Total noise power (set by SNR)

    Reference:
        Middleton, D. "Statistical-Physical Models of Electromagnetic
        Interference", IEEE Trans. EMC, 1977.
    """

    def __init__(self, config: NoiseConfig):
        self.A = config.impulsive_A
        self.Gamma = config.impulsive_Gamma
        self.n_terms = config.n_terms
        self._precompute_weights()

    def _precompute_weights(self):
        """Precompute Poisson mixture weights α_m."""
        m = np.arange(self.n_terms)
        self.alpha = np.exp(-self.A) * (self.A ** m) / factorial(m)
        # Normalize to ensure they sum to 1
        self.alpha /= np.sum(self.alpha)

    def generate(self, n_samples: int, noise_power: float) -> np.ndarray:
        """Generate Middleton Class-A noise samples.

        Args:
            n_samples: Number of complex noise samples.
            noise_power: Total noise power σ² (determined by SNR).

        Returns:
            noise: Complex noise samples (n_samples,)
        """
        noise = np.zeros(n_samples, dtype=complex)

        # For each sample, select a mixture component according to α_m
        component_indices = np.random.choice(
            self.n_terms, size=n_samples, p=self.alpha
        )

        for m in range(self.n_terms):
            mask = component_indices == m
            n_m = np.sum(mask)
            if n_m == 0:
                continue

            # Component variance: σ²_m = σ² · (m/A + Γ) / (1 + Γ)
            sigma2_m = noise_power * (m / self.A + self.Gamma) / (1 + self.Gamma)

            # Generate complex Gaussian: CN(0, σ²_m)
            noise[mask] = np.sqrt(sigma2_m / 2) * (
                np.random.randn(n_m) + 1j * np.random.randn(n_m)
            )

        return noise

    def pdf(self, x: np.ndarray, noise_power: float) -> np.ndarray:
        """Compute the PDF of the Middleton Class-A distribution.

        For real-valued amplitude |x|:
            p(x) = Σ_m α_m · (1/√(2πσ²_m)) · exp(-x²/(2σ²_m))

        Args:
            x: Points at which to evaluate the PDF.
            noise_power: Total noise power σ².

        Returns:
            PDF values at each point x.
        """
        pdf_val = np.zeros_like(x, dtype=float)

        for m in range(self.n_terms):
            sigma2_m = noise_power * (m / self.A + self.Gamma) / (1 + self.Gamma)
            pdf_val += self.alpha[m] * (
                1 / np.sqrt(2 * np.pi * sigma2_m)
            ) * np.exp(-x ** 2 / (2 * sigma2_m))

        return pdf_val

    @staticmethod
    def generate_gaussian(n_samples: int, noise_power: float) -> np.ndarray:
        """Generate standard complex Gaussian noise (AWGN) for comparison.

        Args:
            n_samples: Number of samples.
            noise_power: Noise power σ².

        Returns:
            Complex AWGN samples ~ CN(0, σ²).
        """
        return np.sqrt(noise_power / 2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL ESTIMATORS (Classical)
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelEstimator:
    """Classical Channel Estimation Methods for OFDM.

    Implements LS and LMMSE estimators with interpolation.
    """

    def __init__(self, ofdm_cfg: OFDMConfig):
        self.cfg = ofdm_cfg

    def ls_estimate(self, Y_pilots: np.ndarray,
                    X_pilots: np.ndarray) -> np.ndarray:
        """Least Squares (LS) channel estimation at pilot positions.

        H_LS[k_p] = Y[k_p] / X[k_p]

        This is the simplest estimator but suffers from noise enhancement,
        especially under impulsive noise conditions.

        Args:
            Y_pilots: Received signal at pilot subcarriers.
            X_pilots: Known transmitted pilot symbols.

        Returns:
            H_ls_pilots: LS channel estimates at pilot positions.
        """
        return Y_pilots / X_pilots

    def lmmse_estimate(self, H_ls_pilots: np.ndarray,
                       R_HH_pp: np.ndarray,
                       snr_linear: float) -> np.ndarray:
        """Linear MMSE (LMMSE) channel estimation at pilot positions.

        H_MMSE = R_HpHp · (R_HpHp + (1/SNR)·I)^{-1} · H_LS

        The LMMSE estimator uses channel statistics (autocorrelation)
        to reduce noise, but assumes Gaussian noise statistics.

        Args:
            H_ls_pilots: LS estimates at pilot positions.
            R_HH_pp: Channel autocorrelation at pilot positions (Np × Np).
            snr_linear: Linear SNR value.

        Returns:
            H_mmse_pilots: LMMSE estimates at pilot positions.
        """
        Np = len(H_ls_pilots)
        I = np.eye(Np)
        # LMMSE filter
        W = R_HH_pp @ np.linalg.inv(R_HH_pp + (1 / snr_linear) * I)
        return W @ H_ls_pilots

    def interpolate(self, H_pilots: np.ndarray,
                    method: str = 'linear') -> np.ndarray:
        """Interpolate channel estimates from pilot to all subcarriers.

        Supports linear and cubic spline interpolation.

        Args:
            H_pilots: Channel estimates at pilot positions (Np,).
            method: 'linear' or 'cubic'.

        Returns:
            H_interp: Interpolated channel estimate at all subcarriers (N,).
        """
        pilot_idx = self.cfg.pilot_indices
        all_idx = np.arange(self.cfg.n_subcarriers)

        # Interpolate real and imaginary parts separately
        f_real = interp1d(pilot_idx, H_pilots.real, kind=method,
                          fill_value='extrapolate')
        f_imag = interp1d(pilot_idx, H_pilots.imag, kind=method,
                          fill_value='extrapolate')

        H_interp = f_real(all_idx) + 1j * f_imag(all_idx)
        return H_interp


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE OFDM TRANSMISSION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class OFDMPipeline:
    """End-to-end OFDM transmission and reception pipeline.

    Combines all components: modulation, channel, noise, estimation.
    Used for generating training data and evaluation.
    """

    def __init__(self, ofdm_cfg: OFDMConfig, ch_cfg: ChannelConfig,
                 noise_cfg: NoiseConfig):
        self.ofdm = OFDMSystem(ofdm_cfg)
        self.channel = MultipathChannel(ofdm_cfg, ch_cfg)
        self.impulsive_noise = MiddletonClassA(noise_cfg)
        self.estimator = ChannelEstimator(ofdm_cfg)
        self.ofdm_cfg = ofdm_cfg
        self.ch_cfg = ch_cfg

    def transmit_receive(self, snr_db: float,
                         noise_type: str = 'impulsive') -> dict:
        """Run one complete OFDM transmission-reception cycle.

        Args:
            snr_db: Signal-to-noise ratio in dB.
            noise_type: 'impulsive' (Middleton Class-A) or 'gaussian' (AWGN).

        Returns:
            Dictionary containing all signals and estimates.
        """
        N = self.ofdm_cfg.n_subcarriers

        # ── Transmitter ──
        X_freq, data_symbols, pilot_symbols = self.ofdm.generate_tx_signal()
        x_time = self.ofdm.ofdm_modulate(X_freq)

        # ── Channel ──
        h, H_true = self.channel.generate()
        y_time = self.channel.apply(x_time, h)

        # ── Noise ──
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(y_time) ** 2)
        noise_power = signal_power / snr_linear

        if noise_type == 'impulsive':
            noise = self.impulsive_noise.generate(len(y_time), noise_power)
        else:
            noise = MiddletonClassA.generate_gaussian(len(y_time), noise_power)

        y_time_noisy = y_time + noise

        # ── Receiver ──
        Y_freq = self.ofdm.ofdm_demodulate(y_time_noisy)

        # ── LS Estimation ──
        Y_pilots, X_pilots = self.ofdm.extract_pilots(Y_freq)
        H_ls_pilots = self.estimator.ls_estimate(Y_pilots, X_pilots)
        H_ls_full = self.estimator.interpolate(H_ls_pilots, method='linear')

        # ── LMMSE Estimation ──
        R_HH = self.channel.get_correlation_matrix()
        pilot_idx = self.ofdm_cfg.pilot_indices
        R_HH_pp = R_HH[np.ix_(pilot_idx, pilot_idx)]
        H_mmse_pilots = self.estimator.lmmse_estimate(
            H_ls_pilots, R_HH_pp, snr_linear
        )
        H_mmse_full = self.estimator.interpolate(H_mmse_pilots, method='linear')

        # ── Equalization (Zero-Forcing using true channel for BER reference) ──
        X_eq_true = Y_freq / H_true
        X_eq_ls = Y_freq / (H_ls_full + 1e-10)
        X_eq_mmse = Y_freq / (H_mmse_full + 1e-10)

        return {
            'X_freq': X_freq,
            'x_time': x_time,
            'h': h,
            'H_true': H_true,
            'y_time': y_time,
            'noise': noise[:len(y_time)],
            'y_time_noisy': y_time_noisy,
            'Y_freq': Y_freq,
            'H_ls': H_ls_full,
            'H_mmse': H_mmse_full,
            'X_eq_true': X_eq_true,
            'X_eq_ls': X_eq_ls,
            'X_eq_mmse': X_eq_mmse,
            'data_symbols': data_symbols,
            'snr_db': snr_db,
            'noise_power': noise_power,
        }

    def generate_dataset_sample(self, snr_db: float,
                                noise_type: str = 'impulsive') -> tuple:
        """Generate one (input, target) pair for DL training.

        Input:  LS channel estimate [Re(H_LS); Im(H_LS)] ∈ R^{2N}
        Target: True channel [Re(H); Im(H)] ∈ R^{2N}

        Args:
            snr_db: SNR in dB.
            noise_type: Noise model.

        Returns:
            x_input: LS estimate as real-valued vector (2N,)
            y_target: True channel as real-valued vector (2N,)
        """
        result = self.transmit_receive(snr_db, noise_type)

        H_ls = result['H_ls']
        H_true = result['H_true']

        # Stack real and imaginary parts
        x_input = np.concatenate([H_ls.real, H_ls.imag])
        y_target = np.concatenate([H_true.real, H_true.imag])

        return x_input.astype(np.float32), y_target.astype(np.float32)
