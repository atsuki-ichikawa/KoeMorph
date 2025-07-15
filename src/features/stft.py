"""
Mel-spectrogram feature extraction for audio.

Unit name     : MelSpectrogram
Input         : torch.FloatTensor (B, L)   [16 kHz, -1 ≤ x ≤ 1]
Output        : torch.FloatTensor (B, T, 80)
FPS           : 30
Dependencies  : torchaudio>=2.0
Assumptions   : L divisible by hop_length
Failure modes : CentFreq mismatch, NaN, long-time
Test cases    : test_shape, test_reconstruction_snr, test_grad
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.nn import functional as F


class MelSpectrogramExtractor(nn.Module):
    """
    Mel-spectrogram feature extractor optimized for 30 FPS output.

    Converts raw audio to log-mel spectrograms with temporal resolution
    matching the target blendshape frame rate.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        n_fft: int = 512,
        n_mels: int = 80,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = True,
        center: bool = True,
        pad_mode: str = "reflect",
        eps: float = 1e-8,
    ):
        """
        Initialize mel-spectrogram extractor.

        Args:
            sample_rate: Audio sample rate in Hz
            target_fps: Target output frame rate (frames per second)
            n_fft: FFT window size
            n_mels: Number of mel filter banks
            f_min: Minimum frequency for mel scale
            f_max: Maximum frequency for mel scale (defaults to sr/2)
            power: Power to raise the magnitude spectrogram (1.0 for magnitude, 2.0 for power)
            normalized: Whether to normalize mel filter banks
            center: Whether to center the FFT window
            pad_mode: Padding mode for STFT
            eps: Small value to avoid log(0)
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power
        self.eps = eps

        # Calculate hop length for target FPS
        # hop_length = sample_rate / target_fps
        self.hop_length = int(sample_rate / target_fps)
        self.win_length = n_fft

        # Validate parameters
        if self.hop_length <= 0:
            raise ValueError(
                f"Invalid hop_length {self.hop_length} for sr={sample_rate}, fps={target_fps}"
            )

        # Create mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
        )

        # Register mel scale parameters for inspection
        self.register_buffer("mel_scale", self.mel_transform.mel_scale.fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram from waveform.

        Args:
            waveform: Input waveform of shape (B, L) or (L,)

        Returns:
            Log mel-spectrogram of shape (B, T, n_mels)
        """
        # Handle single sample input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2:
            raise ValueError(f"Expected 1D or 2D input, got {waveform.dim()}D")

        # Extract mel-spectrogram
        # Output shape: (B, n_mels, T)
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale and add small epsilon to avoid log(0)
        log_mel = torch.log(mel_spec + self.eps)

        # Transpose to (B, T, n_mels) for sequence modeling
        log_mel = log_mel.transpose(1, 2)

        return log_mel

    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output sequence length for given input length.

        Args:
            input_length: Length of input waveform

        Returns:
            Length of output mel-spectrogram sequence
        """
        # Account for center padding
        if self.mel_transform.center:
            input_length += 2 * (self.n_fft // 2)

        return (input_length - self.n_fft) // self.hop_length + 1

    def get_time_axis(self, seq_length: int) -> torch.Tensor:
        """
        Get time axis for mel-spectrogram frames.

        Args:
            seq_length: Length of mel-spectrogram sequence

        Returns:
            Time values in seconds for each frame
        """
        frame_indices = torch.arange(seq_length, dtype=torch.float32)
        time_seconds = frame_indices * self.hop_length / self.sample_rate

        return time_seconds


class InverseMelSpectrogram(nn.Module):
    """
    Inverse mel-spectrogram transform for reconstruction testing.

    Uses Griffin-Lim algorithm to reconstruct waveform from mel-spectrogram.
    Primarily used for testing and validation.
    """

    def __init__(
        self,
        mel_extractor: MelSpectrogramExtractor,
        n_iter: int = 32,
        momentum: float = 0.99,
        length: Optional[int] = None,
    ):
        """
        Initialize inverse transform.

        Args:
            mel_extractor: Forward mel-spectrogram extractor
            n_iter: Number of Griffin-Lim iterations
            momentum: Momentum for Griffin-Lim
            length: Target output length (None for auto)
        """
        super().__init__()

        self.mel_extractor = mel_extractor
        self.n_iter = n_iter
        self.momentum = momentum
        self.length = length

        # Create inverse mel scale transform
        self.inverse_mel = T.InverseMelScale(
            sample_rate=mel_extractor.sample_rate,
            n_stft=mel_extractor.n_fft // 2 + 1,
            n_mels=mel_extractor.n_mels,
            f_min=mel_extractor.f_min,
            f_max=mel_extractor.f_max,
        )

        # Create Griffin-Lim transform
        self.griffin_lim = T.GriffinLim(
            n_fft=mel_extractor.n_fft,
            n_iter=n_iter,
            win_length=mel_extractor.win_length,
            hop_length=mel_extractor.hop_length,
            power=mel_extractor.power,
            momentum=momentum,
            length=length,
        )

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from log mel-spectrogram.

        Args:
            log_mel: Log mel-spectrogram of shape (B, T, n_mels)

        Returns:
            Reconstructed waveform of shape (B, L)
        """
        # Convert from log scale
        mel_spec = torch.exp(log_mel - self.mel_extractor.eps)

        # Transpose to (B, n_mels, T)
        mel_spec = mel_spec.transpose(1, 2)

        # Convert mel to linear spectrogram
        linear_spec = self.inverse_mel(mel_spec)

        # Reconstruct waveform using Griffin-Lim
        waveform = self.griffin_lim(linear_spec)

        return waveform


def compute_reconstruction_snr(
    original: torch.Tensor, reconstructed: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio between original and reconstructed signals.

    Args:
        original: Original signal
        reconstructed: Reconstructed signal
        eps: Small value for numerical stability

    Returns:
        SNR in dB
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[..., :min_len]
    reconstructed = reconstructed[..., :min_len]

    # Compute power of signal and noise
    signal_power = torch.mean(original**2, dim=-1)
    noise_power = torch.mean((original - reconstructed) ** 2, dim=-1)

    # Compute SNR in dB
    snr_db = 10 * torch.log10((signal_power + eps) / (noise_power + eps))

    return snr_db


def validate_mel_parameters(
    sample_rate: int,
    target_fps: float,
    n_fft: int,
    n_mels: int,
    f_max: Optional[float] = None,
) -> dict:
    """
    Validate mel-spectrogram parameters and return diagnostics.

    Args:
        sample_rate: Audio sample rate
        target_fps: Target frame rate
        n_fft: FFT size
        n_mels: Number of mel bins
        f_max: Maximum frequency

    Returns:
        Dictionary with validation results and diagnostics
    """
    results = {"valid": True, "warnings": [], "info": {}}

    # Calculate hop length
    hop_length = int(sample_rate / target_fps)
    results["info"]["hop_length"] = hop_length
    results["info"]["actual_fps"] = sample_rate / hop_length

    # Check hop length validity
    if hop_length <= 0:
        results["valid"] = False
        results["warnings"].append(f"Invalid hop_length {hop_length}")

    # Check frequency resolution
    freq_resolution = sample_rate / n_fft
    results["info"]["freq_resolution_hz"] = freq_resolution

    # Check temporal resolution
    temporal_resolution = hop_length / sample_rate
    results["info"]["temporal_resolution_s"] = temporal_resolution

    # Check Nyquist frequency
    nyquist = sample_rate / 2
    if f_max and f_max > nyquist:
        results["warnings"].append(f"f_max {f_max} > Nyquist {nyquist}")

    # Check for reasonable parameters
    if n_fft < hop_length:
        results["warnings"].append(
            f"n_fft {n_fft} < hop_length {hop_length} may cause artifacts"
        )

    if abs(results["info"]["actual_fps"] - target_fps) > 0.1:
        results["warnings"].append(
            f"Actual FPS {results['info']['actual_fps']:.2f} differs from target {target_fps}"
        )

    return results
