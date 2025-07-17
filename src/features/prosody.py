"""
Prosodic feature extraction (F0, log energy, VAD).

Unit name     : ProsodyExtractor
Input         : torch.FloatTensor (B, L)   [16 kHz audio]
Output        : torch.FloatTensor (B, T, 4) [F0, logE, VAD, voicing]
Dependencies  : librosa, scipy
Assumptions   : Synchronized with mel features at target FPS
Failure modes : F0 tracking errors, silent segments
Test cases    : test_shape, test_f0_range, test_vad_accuracy
"""

import warnings
from typing import Optional, Tuple

import librosa
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from scipy.interpolate import interp1d


class ProsodyExtractor(nn.Module):
    """
    Prosodic feature extractor for speech signals.

    Extracts fundamental frequency (F0), log energy, voice activity detection (VAD),
    and voicing probability with temporal alignment to target FPS.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        frame_length: float = 0.025,  # 25ms
        frame_shift: float = 0.010,  # 10ms
        f0_min: float = 80.0,
        f0_max: float = 400.0,
        energy_floor: float = 1e-8,
        vad_threshold: float = 0.01,
        interpolate_unvoiced: bool = True,
    ):
        """
        Initialize prosody extractor.

        Args:
            sample_rate: Audio sample rate in Hz
            target_fps: Target output frame rate
            frame_length: Frame length in seconds for analysis
            frame_shift: Frame shift in seconds for analysis
            f0_min: Minimum F0 for pitch tracking (Hz)
            f0_max: Maximum F0 for pitch tracking (Hz)
            energy_floor: Floor value for energy computation
            vad_threshold: Threshold for voice activity detection
            interpolate_unvoiced: Whether to interpolate F0 in unvoiced regions
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.energy_floor = energy_floor
        self.vad_threshold = vad_threshold
        self.interpolate_unvoiced = interpolate_unvoiced

        # Convert time parameters to samples
        self.frame_length_samples = int(frame_length * sample_rate)
        self.frame_shift_samples = int(frame_shift * sample_rate)

        # Target frame shift for output alignment
        self.target_frame_shift = int(sample_rate / target_fps)

        # Pre-emphasis filter coefficients
        self.register_buffer("preemph_coeff", torch.tensor([1.0, -0.97]))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract prosodic features from waveform.

        Args:
            waveform: Input waveform of shape (B, L) or (L,)

        Returns:
            Prosodic features of shape (B, T, 4) where features are:
            [F0 (Hz), log_energy, VAD, voicing_prob]
        """
        # Handle single sample input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2:
            raise ValueError(f"Expected 1D or 2D input, got {waveform.dim()}D")

        batch_size = waveform.shape[0]
        features_list = []

        # Process each sample in batch
        for i in range(batch_size):
            audio = waveform[i].detach().cpu().numpy()
            features = self._extract_prosody_single(audio)
            features_list.append(features)

        # Stack and convert to tensor
        features = torch.stack(features_list, dim=0)

        return features

    def _extract_prosody_single(self, audio: np.ndarray) -> torch.Tensor:
        """Extract prosodic features for a single audio signal."""
        # Apply pre-emphasis
        audio_preemph = scipy.signal.lfilter([1.0, -0.97], [1.0], audio)

        # Extract F0 using librosa (PYIN algorithm)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_preemph,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            frame_length=self.frame_length_samples,
            hop_length=self.frame_shift_samples,
            fill_na=0.0,  # Fill unvoiced with 0
        )

        # Handle NaN values
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_probs = np.nan_to_num(voiced_probs, nan=0.0)

        # Extract log energy
        log_energy = self._extract_log_energy(audio_preemph)

        # Extract VAD
        vad = self._extract_vad(audio_preemph, log_energy)

        # Interpolate F0 in unvoiced regions if requested
        if self.interpolate_unvoiced:
            f0 = self._interpolate_f0(f0, voiced_flag)

        # Ensure all features have the same length
        min_len = min(len(f0), len(log_energy), len(vad), len(voiced_probs))
        f0 = f0[:min_len]
        log_energy = log_energy[:min_len]
        vad = vad[:min_len]
        voiced_probs = voiced_probs[:min_len]

        # Resample to target FPS
        f0_resampled = self._resample_to_target_fps(f0, len(audio))
        energy_resampled = self._resample_to_target_fps(log_energy, len(audio))
        vad_resampled = self._resample_to_target_fps(vad, len(audio))
        voiced_resampled = self._resample_to_target_fps(voiced_probs, len(audio))

        # Stack features
        features = np.stack(
            [f0_resampled, energy_resampled, vad_resampled, voiced_resampled], axis=1
        )

        # Get device from buffers or parameters
        try:
            device = next(self.parameters()).device
        except StopIteration:
            try:
                device = next(self.buffers()).device
            except StopIteration:
                device = torch.device('cpu')  # fallback
        return torch.from_numpy(features.astype(np.float32)).to(device)

    def _extract_log_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract log energy from audio signal."""
        # Frame the audio
        frames = librosa.util.frame(
            audio,
            frame_length=self.frame_length_samples,
            hop_length=self.frame_shift_samples,
            axis=0,
        )

        # Compute energy (RMS)
        energy = np.sqrt(np.mean(frames**2, axis=0))

        # Convert to log scale
        log_energy = np.log(energy + self.energy_floor)

        return log_energy

    def _extract_vad(self, audio: np.ndarray, log_energy: np.ndarray) -> np.ndarray:
        """Extract voice activity detection from audio signal."""
        # Simple energy-based VAD
        # Normalize log energy to [0, 1] range
        energy_norm = (log_energy - log_energy.min()) / (
            log_energy.max() - log_energy.min() + 1e-8
        )

        # Apply threshold
        vad = (energy_norm > self.vad_threshold).astype(np.float32)

        # Apply median filter to smooth VAD decisions
        if len(vad) > 5:
            vad = scipy.signal.medfilt(vad, kernel_size=5)

        return vad

    def _interpolate_f0(self, f0: np.ndarray, voiced_flag: np.ndarray) -> np.ndarray:
        """Interpolate F0 values in unvoiced regions."""
        if not np.any(voiced_flag):
            return f0  # No voiced segments to interpolate from

        # Get voiced F0 values
        voiced_indices = np.where(voiced_flag)[0]
        voiced_f0 = f0[voiced_indices]

        if len(voiced_indices) < 2:
            return f0  # Need at least 2 points for interpolation

        # Interpolate
        interp_func = interp1d(
            voiced_indices,
            voiced_f0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Apply interpolation only to unvoiced regions
        f0_interp = f0.copy()
        unvoiced_indices = np.where(~voiced_flag)[0]

        # Only interpolate between voiced segments (not at edges)
        for idx in unvoiced_indices:
            if voiced_indices[0] < idx < voiced_indices[-1]:
                f0_interp[idx] = interp_func(idx)

        return f0_interp

    def _resample_to_target_fps(
        self, features: np.ndarray, audio_length: int
    ) -> np.ndarray:
        """Resample features to target FPS."""
        # Current time axis (based on frame shift)
        current_frames = len(features)
        current_times = np.arange(current_frames) * self.frame_shift

        # Target time axis (based on target FPS)
        audio_duration = audio_length / self.sample_rate
        target_frames = int(audio_duration * self.target_fps)
        target_times = np.arange(target_frames) / self.target_fps

        if target_frames == 0:
            return np.array([])

        # Interpolate to target time grid
        if current_frames > 1:
            interp_func = interp1d(
                current_times,
                features,
                kind="linear",
                bounds_error=False,
                fill_value=(features[0], features[-1]),
            )
            features_resampled = interp_func(target_times)
        else:
            # If only one frame, repeat it
            features_resampled = np.full(
                target_frames, features[0] if current_frames > 0 else 0.0
            )

        return features_resampled

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length for given input length."""
        audio_duration = input_length / self.sample_rate
        return int(audio_duration * self.target_fps)


class ProsodyNormalizer(nn.Module):
    """
    Normalizer for prosodic features with speaker-adaptive statistics.

    Normalizes F0, energy, and other prosodic features to improve
    model generalization across different speakers.
    """

    def __init__(
        self,
        f0_log_transform: bool = True,
        f0_mean: float = 150.0,
        f0_std: float = 50.0,
        energy_mean: float = -5.0,
        energy_std: float = 2.0,
        eps: float = 1e-8,
    ):
        """
        Initialize prosody normalizer.

        Args:
            f0_log_transform: Whether to apply log transform to F0
            f0_mean: Mean F0 for normalization (Hz)
            f0_std: Standard deviation of F0 for normalization (Hz)
            energy_mean: Mean log energy for normalization
            energy_std: Standard deviation of log energy for normalization
            eps: Small value for numerical stability
        """
        super().__init__()

        self.f0_log_transform = f0_log_transform
        self.eps = eps

        # Register normalization parameters
        self.register_buffer("f0_mean", torch.tensor(f0_mean))
        self.register_buffer("f0_std", torch.tensor(f0_std))
        self.register_buffer("energy_mean", torch.tensor(energy_mean))
        self.register_buffer("energy_std", torch.tensor(energy_std))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize prosodic features.

        Args:
            features: Input features of shape (B, T, 4) [F0, logE, VAD, voicing]

        Returns:
            Normalized features of same shape
        """
        features_norm = features.clone()

        # Normalize F0 (channel 0)
        f0 = features_norm[..., 0]
        voiced_mask = f0 > 0  # Only normalize voiced frames

        if voiced_mask.any():
            if self.f0_log_transform:
                # Apply log transform to voiced F0
                f0_voiced = f0[voiced_mask]
                f0_log = torch.log(f0_voiced + self.eps)
                f0_log_norm = (f0_log - torch.log(self.f0_mean + self.eps)) / (
                    torch.log(self.f0_std + self.eps) + self.eps
                )
                features_norm[..., 0] = torch.where(voiced_mask, f0_log_norm, f0)
            else:
                # Linear normalization
                f0_norm = (f0 - self.f0_mean) / (self.f0_std + self.eps)
                features_norm[..., 0] = torch.where(voiced_mask, f0_norm, f0)

        # Normalize log energy (channel 1)
        energy = features_norm[..., 1]
        energy_norm = (energy - self.energy_mean) / (self.energy_std + self.eps)
        features_norm[..., 1] = energy_norm

        # VAD and voicing probability don't need normalization (already in [0,1])

        return features_norm

    def denormalize(self, features_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize prosodic features back to original scale.

        Args:
            features_norm: Normalized features of shape (B, T, 4)

        Returns:
            Denormalized features
        """
        features = features_norm.clone()

        # Denormalize F0
        f0_norm = features[..., 0]
        voiced_mask = features[..., 3] > 0.5  # Use voicing probability

        if voiced_mask.any():
            if self.f0_log_transform:
                f0_log_denorm = f0_norm * (
                    torch.log(self.f0_std + self.eps) + self.eps
                ) + torch.log(self.f0_mean + self.eps)
                f0_denorm = torch.exp(f0_log_denorm) - self.eps
                features[..., 0] = torch.where(
                    voiced_mask, f0_denorm, torch.zeros_like(f0_norm)
                )
            else:
                f0_denorm = f0_norm * (self.f0_std + self.eps) + self.f0_mean
                features[..., 0] = torch.where(
                    voiced_mask, f0_denorm, torch.zeros_like(f0_norm)
                )

        # Denormalize energy
        energy_norm = features[..., 1]
        energy_denorm = energy_norm * (self.energy_std + self.eps) + self.energy_mean
        features[..., 1] = energy_denorm

        return features


def validate_prosody_features(features: torch.Tensor) -> dict:
    """
    Validate prosodic features for quality and consistency.

    Args:
        features: Prosodic features of shape (B, T, 4)

    Returns:
        Dictionary with validation results
    """
    results = {"valid": True, "warnings": [], "stats": {}}

    if features.dim() != 3 or features.shape[2] != 4:
        results["valid"] = False
        results["warnings"].append(f"Expected shape (B, T, 4), got {features.shape}")
        return results

    f0 = features[..., 0]
    energy = features[..., 1]
    vad = features[..., 2]
    voicing = features[..., 3]

    # F0 statistics
    voiced_frames = f0 > 0
    if voiced_frames.any():
        f0_voiced = f0[voiced_frames]
        results["stats"]["f0_mean"] = f0_voiced.mean().item()
        results["stats"]["f0_std"] = f0_voiced.std().item()
        results["stats"]["f0_min"] = f0_voiced.min().item()
        results["stats"]["f0_max"] = f0_voiced.max().item()

        # Check for reasonable F0 range
        if results["stats"]["f0_min"] < 50 or results["stats"]["f0_max"] > 500:
            results["warnings"].append(
                f"F0 range unusual: {results['stats']['f0_min']:.1f}-{results['stats']['f0_max']:.1f} Hz"
            )
    else:
        results["warnings"].append("No voiced frames detected")

    # Energy statistics
    results["stats"]["energy_mean"] = energy.mean().item()
    results["stats"]["energy_std"] = energy.std().item()

    # VAD statistics
    vad_rate = vad.mean().item()
    results["stats"]["vad_rate"] = vad_rate

    if vad_rate < 0.1:
        results["warnings"].append(f"Very low VAD rate: {vad_rate:.3f}")
    elif vad_rate > 0.9:
        results["warnings"].append(f"Very high VAD rate: {vad_rate:.3f}")

    # Voicing statistics
    voicing_rate = voicing.mean().item()
    results["stats"]["voicing_rate"] = voicing_rate

    # Check for NaN or infinite values
    if torch.isnan(features).any():
        results["valid"] = False
        results["warnings"].append("NaN values detected")

    if torch.isinf(features).any():
        results["valid"] = False
        results["warnings"].append("Infinite values detected")

    return results
