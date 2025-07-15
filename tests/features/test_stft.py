"""Tests for features.stft module."""

import numpy as np
import pytest
import torch

from src.features.stft import (
    InverseMelSpectrogram,
    MelSpectrogramExtractor,
    compute_reconstruction_snr,
    validate_mel_parameters,
)


class TestMelSpectrogramExtractor:
    """Test mel-spectrogram extraction functionality."""

    def test_extractor_creation(self):
        """Test basic extractor creation."""
        extractor = MelSpectrogramExtractor()

        assert extractor.sample_rate == 16000
        assert extractor.target_fps == 30.0
        assert extractor.n_mels == 80
        assert extractor.hop_length == 16000 // 30  # ~533

    def test_forward_shape_single_sample(self):
        """Test forward pass shape with single sample."""
        extractor = MelSpectrogramExtractor(sample_rate=16000, target_fps=30)

        # 1 second of audio
        audio = torch.randn(16000)

        mel_spec = extractor(audio)

        # Should be (1, T, 80) since we unsqueeze batch dim
        assert mel_spec.dim() == 3
        assert mel_spec.shape[0] == 1  # Batch size
        assert mel_spec.shape[2] == 80  # n_mels

        # Check approximate time dimension (30 FPS for 1 second ≈ 30 frames)
        expected_frames = extractor.get_output_length(16000)
        assert mel_spec.shape[1] == expected_frames

    def test_forward_shape_batch(self):
        """Test forward pass shape with batch."""
        extractor = MelSpectrogramExtractor()

        # Batch of 3 samples, each 0.5 seconds
        batch_audio = torch.randn(3, 8000)

        mel_spec = extractor(batch_audio)

        assert mel_spec.shape[0] == 3  # Batch size
        assert mel_spec.shape[2] == 80  # n_mels

        # Check time dimension
        expected_frames = extractor.get_output_length(8000)
        assert mel_spec.shape[1] == expected_frames

    def test_output_length_calculation(self):
        """Test output length calculation."""
        extractor = MelSpectrogramExtractor(sample_rate=16000, target_fps=30)

        # Test various input lengths
        test_lengths = [8000, 16000, 32000]  # 0.5s, 1s, 2s

        for input_len in test_lengths:
            output_len = extractor.get_output_length(input_len)

            # Should be approximately input_len * fps / sample_rate
            expected_len = int(input_len * 30 / 16000)

            # Allow some tolerance due to windowing
            assert abs(output_len - expected_len) <= 2

    def test_time_axis(self):
        """Test time axis calculation."""
        extractor = MelSpectrogramExtractor(sample_rate=16000, target_fps=30)

        seq_length = 30  # 30 frames
        time_axis = extractor.get_time_axis(seq_length)

        assert len(time_axis) == seq_length
        assert time_axis[0] == 0.0  # Starts at 0

        # Check time spacing (should be ~1/30 seconds)
        time_diff = time_axis[1] - time_axis[0]
        expected_diff = extractor.hop_length / extractor.sample_rate
        assert abs(time_diff - expected_diff) < 1e-6

    def test_gradient_flow(self):
        """Test that gradients flow through the extractor."""
        extractor = MelSpectrogramExtractor()

        # Input with gradient tracking
        audio = torch.randn(1000, requires_grad=True)

        mel_spec = extractor(audio)
        loss = mel_spec.sum()
        loss.backward()

        # Check that gradients exist
        assert audio.grad is not None
        assert not torch.all(audio.grad == 0)

    def test_invalid_input_dimension(self):
        """Test error handling for invalid input dimensions."""
        extractor = MelSpectrogramExtractor()

        # 3D input should raise error
        invalid_input = torch.randn(2, 1000, 80)

        with pytest.raises(ValueError, match="Expected 1D or 2D input"):
            extractor(invalid_input)

    def test_custom_parameters(self):
        """Test extractor with custom parameters."""
        extractor = MelSpectrogramExtractor(
            sample_rate=22050, target_fps=25.0, n_mels=128, n_fft=1024
        )

        assert extractor.sample_rate == 22050
        assert extractor.target_fps == 25.0
        assert extractor.n_mels == 128
        assert extractor.n_fft == 1024
        assert extractor.hop_length == 22050 // 25  # 882

        # Test with audio
        audio = torch.randn(22050)  # 1 second
        mel_spec = extractor(audio)

        assert mel_spec.shape[2] == 128  # Custom n_mels

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Zero or negative sample rate should cause issues
        with pytest.raises(ValueError):
            MelSpectrogramExtractor(
                sample_rate=100, target_fps=200
            )  # hop_length would be 0


class TestInverseMelSpectrogram:
    """Test inverse mel-spectrogram reconstruction."""

    @pytest.fixture
    def mel_extractor(self):
        """Create mel extractor for testing."""
        return MelSpectrogramExtractor(
            sample_rate=16000, target_fps=30, n_fft=512, n_mels=80
        )

    def test_inverse_creation(self, mel_extractor):
        """Test inverse transform creation."""
        inverse = InverseMelSpectrogram(mel_extractor)

        assert inverse.mel_extractor is mel_extractor
        assert inverse.n_iter == 32
        assert inverse.momentum == 0.99

    def test_reconstruction_basic(self, mel_extractor):
        """Test basic reconstruction functionality."""
        inverse = InverseMelSpectrogram(
            mel_extractor, n_iter=8
        )  # Fewer iterations for speed

        # Generate test audio
        original_audio = torch.randn(8000)  # 0.5 seconds

        # Extract mel-spectrogram
        mel_spec = mel_extractor(original_audio)

        # Reconstruct
        reconstructed = inverse(mel_spec)

        # Check shape
        assert reconstructed.dim() == 2  # (B, L)
        assert reconstructed.shape[0] == 1  # Batch size

        # Check approximate length (Griffin-Lim may change length slightly)
        assert abs(reconstructed.shape[1] - len(original_audio)) < 1000

    def test_reconstruction_snr(self, mel_extractor):
        """Test reconstruction quality using SNR."""
        inverse = InverseMelSpectrogram(mel_extractor, n_iter=16)

        # Generate clean test signal (sine wave)
        sample_rate = 16000
        duration = 1.0
        freq = 440  # A4
        t = torch.linspace(0, duration, int(sample_rate * duration))
        original_audio = 0.5 * torch.sin(2 * torch.pi * freq * t)

        # Extract and reconstruct
        mel_spec = mel_extractor(original_audio)
        reconstructed = inverse(mel_spec).squeeze(0)

        # Compute SNR
        snr = compute_reconstruction_snr(original_audio, reconstructed)

        # Should achieve reasonable SNR (>10 dB for sine wave)
        assert snr > 10.0


class TestReconstructionSNR:
    """Test SNR computation utilities."""

    def test_snr_identical_signals(self):
        """Test SNR for identical signals."""
        signal = torch.randn(1000)
        snr = compute_reconstruction_snr(signal, signal)

        # SNR should be very high for identical signals
        assert snr > 80  # Very high SNR

    def test_snr_noisy_signal(self):
        """Test SNR for noisy signal."""
        signal = torch.randn(1000)
        noise = 0.1 * torch.randn(1000)
        noisy_signal = signal + noise

        snr = compute_reconstruction_snr(signal, noisy_signal)

        # Should detect the noise level
        assert 15 < snr < 25  # Reasonable SNR range for 10% noise

    def test_snr_different_lengths(self):
        """Test SNR computation with different length signals."""
        signal1 = torch.randn(1000)
        signal2 = torch.randn(800)  # Shorter

        snr = compute_reconstruction_snr(signal1, signal2)

        # Should handle length difference gracefully
        assert isinstance(snr, torch.Tensor)
        assert snr.numel() == 1

    def test_snr_batch(self):
        """Test SNR computation with batch inputs."""
        batch_size = 3
        signal_length = 500

        original = torch.randn(batch_size, signal_length)
        reconstructed = original + 0.05 * torch.randn(batch_size, signal_length)

        snr = compute_reconstruction_snr(original, reconstructed)

        assert snr.shape == (batch_size,)
        assert torch.all(snr > 20)  # Should be high SNR for low noise


class TestParameterValidation:
    """Test parameter validation utilities."""

    def test_valid_parameters(self):
        """Test validation with valid parameters."""
        results = validate_mel_parameters(
            sample_rate=16000, target_fps=30.0, n_fft=512, n_mels=80
        )

        assert results["valid"] is True
        assert "hop_length" in results["info"]
        assert "actual_fps" in results["info"]
        assert abs(results["info"]["actual_fps"] - 30.0) < 0.1

    def test_invalid_sample_rate(self):
        """Test validation with invalid sample rate."""
        results = validate_mel_parameters(
            sample_rate=100, target_fps=30.0, n_fft=512, n_mels=80  # Very low
        )

        assert results["valid"] is False
        assert any("Invalid hop_length" in w for w in results["warnings"])

    def test_high_fmax(self):
        """Test validation with f_max above Nyquist."""
        results = validate_mel_parameters(
            sample_rate=16000,
            target_fps=30.0,
            n_fft=512,
            n_mels=80,
            f_max=10000,  # Above Nyquist (8000)
        )

        assert any("Nyquist" in w for w in results["warnings"])

    def test_nfft_hop_mismatch(self):
        """Test validation with n_fft < hop_length."""
        results = validate_mel_parameters(
            sample_rate=16000, target_fps=10.0, n_fft=512, n_mels=80  # Large hop_length
        )

        # hop_length would be 1600, which is > n_fft (512)
        assert any("hop_length" in w for w in results["warnings"])

    def test_fps_mismatch(self):
        """Test validation with FPS mismatch."""
        results = validate_mel_parameters(
            sample_rate=16000, target_fps=29.7, n_fft=512, n_mels=80  # Unusual FPS
        )

        # Should warn about FPS difference
        assert any("differs from target" in w for w in results["warnings"])

    def test_parameter_info(self):
        """Test that parameter info is correctly computed."""
        results = validate_mel_parameters(
            sample_rate=22050, target_fps=25.0, n_fft=1024, n_mels=128
        )

        expected_hop = 22050 // 25
        assert results["info"]["hop_length"] == expected_hop
        assert abs(results["info"]["actual_fps"] - 25.0) < 0.1
        assert results["info"]["freq_resolution_hz"] == 22050 / 1024
        assert results["info"]["temporal_resolution_s"] == expected_hop / 22050
