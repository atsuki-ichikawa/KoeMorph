"""Tests for features.prosody module."""

import numpy as np
import pytest
import torch

from src.features.prosody import (ProsodyExtractor, ProsodyNormalizer,
                                  validate_prosody_features)


class TestProsodyExtractor:
    """Test prosodic feature extraction functionality."""

    def test_extractor_creation(self):
        """Test basic extractor creation."""
        extractor = ProsodyExtractor()

        assert extractor.sample_rate == 16000
        assert extractor.target_fps == 30.0
        assert extractor.f0_min == 80.0
        assert extractor.f0_max == 400.0

    def test_forward_shape_single_sample(self):
        """Test forward pass shape with single sample."""
        extractor = ProsodyExtractor(sample_rate=16000, target_fps=30)

        # 1 second of audio with a sine wave (should be voiced)
        t = torch.linspace(0, 1, 16000)
        audio = 0.5 * torch.sin(2 * torch.pi * 220 * t)  # 220 Hz tone

        features = extractor(audio)

        # Should be (1, T, 4) since we unsqueeze batch dim
        assert features.dim() == 3
        assert features.shape[0] == 1  # Batch size
        assert features.shape[2] == 4  # [F0, logE, VAD, voicing]

        # Check approximate time dimension (30 FPS for 1 second ≈ 30 frames)
        expected_frames = extractor.get_output_length(16000)
        assert features.shape[1] == expected_frames

    def test_forward_shape_batch(self):
        """Test forward pass shape with batch."""
        extractor = ProsodyExtractor()

        # Batch of 2 samples, each 0.5 seconds
        batch_audio = torch.randn(2, 8000)

        features = extractor(batch_audio)

        assert features.shape[0] == 2  # Batch size
        assert features.shape[2] == 4  # Feature dimension

        # Check time dimension
        expected_frames = extractor.get_output_length(8000)
        assert features.shape[1] == expected_frames

    def test_sine_wave_f0_detection(self):
        """Test F0 detection on a clean sine wave."""
        extractor = ProsodyExtractor()

        # Generate 1-second sine wave at 220 Hz
        sample_rate = 16000
        t = torch.linspace(0, 1, sample_rate)
        freq = 220.0
        audio = 0.5 * torch.sin(2 * torch.pi * freq * t)

        features = extractor(audio)

        # Extract F0 and voicing
        f0 = features[0, :, 0]  # First batch, all time, F0 channel
        voicing = features[0, :, 3]  # Voicing probability

        # Should detect some voiced frames
        voiced_frames = voicing > 0.5
        assert voiced_frames.any(), "No voiced frames detected in sine wave"

        # F0 should be close to 220 Hz in voiced frames
        if voiced_frames.any():
            f0_voiced = f0[voiced_frames]
            f0_mean = f0_voiced.mean()

            # Allow reasonable tolerance (±20 Hz)
            assert (
                abs(f0_mean - freq) < 20
            ), f"F0 detection error: {f0_mean:.1f} vs {freq:.1f} Hz"

    def test_silent_audio_vad(self):
        """Test VAD on silent audio."""
        extractor = ProsodyExtractor()

        # Generate silent audio
        audio = torch.zeros(8000)  # 0.5 seconds of silence

        features = extractor(audio)

        # VAD should be mostly inactive
        vad = features[0, :, 2]
        vad_rate = vad.mean()

        assert vad_rate < 0.3, f"VAD too active on silent audio: {vad_rate:.3f}"

    def test_noisy_audio_robustness(self):
        """Test robustness to noisy audio."""
        extractor = ProsodyExtractor()

        # Generate noisy sine wave
        t = torch.linspace(0, 1, 16000)
        clean_signal = 0.5 * torch.sin(2 * torch.pi * 150 * t)
        noise = 0.1 * torch.randn(16000)
        noisy_audio = clean_signal + noise

        features = extractor(noisy_audio)

        # Should still detect some structure
        f0 = features[0, :, 0]
        voicing = features[0, :, 3]
        vad = features[0, :, 2]

        # Basic sanity checks
        assert not torch.isnan(features).any(), "NaN values in features"
        assert not torch.isinf(features).any(), "Infinite values in features"
        assert vad.max() <= 1.0 and vad.min() >= 0.0, "VAD out of range"
        assert (
            voicing.max() <= 1.0 and voicing.min() >= 0.0
        ), "Voicing prob out of range"

    def test_output_length_calculation(self):
        """Test output length calculation."""
        extractor = ProsodyExtractor(sample_rate=16000, target_fps=30)

        # Test various input lengths
        test_lengths = [8000, 16000, 32000]  # 0.5s, 1s, 2s

        for input_len in test_lengths:
            output_len = extractor.get_output_length(input_len)

            # Should be approximately input_len * fps / sample_rate
            expected_len = int(input_len * 30 / 16000)

            # Allow small tolerance
            assert abs(output_len - expected_len) <= 1

    def test_custom_parameters(self):
        """Test extractor with custom parameters."""
        extractor = ProsodyExtractor(
            sample_rate=22050,
            target_fps=25.0,
            f0_min=100.0,
            f0_max=300.0,
            frame_length=0.05,  # 50ms frames
            frame_shift=0.02,  # 20ms shift
        )

        assert extractor.sample_rate == 22050
        assert extractor.target_fps == 25.0
        assert extractor.f0_min == 100.0
        assert extractor.f0_max == 300.0

        # Test with audio
        audio = torch.randn(22050)  # 1 second
        features = extractor(audio)

        expected_frames = int(22050 * 25.0 / 22050)  # 25 frames
        assert features.shape[1] == expected_frames

    def test_invalid_input_dimension(self):
        """Test error handling for invalid input dimensions."""
        extractor = ProsodyExtractor()

        # 3D input should raise error
        invalid_input = torch.randn(2, 1000, 80)

        with pytest.raises(ValueError, match="Expected 1D or 2D input"):
            extractor(invalid_input)


class TestProsodyNormalizer:
    """Test prosody normalization functionality."""

    @pytest.fixture
    def sample_features(self):
        """Create sample prosodic features for testing."""
        batch_size, seq_len = 2, 30

        # Create realistic prosodic features
        features = torch.zeros(batch_size, seq_len, 4)

        # F0: 120-200 Hz with some unvoiced frames (0)
        f0 = torch.linspace(120, 200, seq_len).unsqueeze(0).repeat(batch_size, 1)
        f0[:, 10:15] = 0  # Unvoiced segment
        features[:, :, 0] = f0

        # Log energy: around -3 to -8
        features[:, :, 1] = (
            torch.linspace(-8, -3, seq_len).unsqueeze(0).repeat(batch_size, 1)
        )

        # VAD: mostly voiced except unvoiced segment
        vad = torch.ones(batch_size, seq_len)
        vad[:, 10:15] = 0
        features[:, :, 2] = vad

        # Voicing probability: similar to VAD but continuous
        voicing = torch.ones(batch_size, seq_len) * 0.9
        voicing[:, 10:15] = 0.1  # Low probability for unvoiced
        features[:, :, 3] = voicing

        return features

    def test_normalizer_creation(self):
        """Test normalizer creation."""
        normalizer = ProsodyNormalizer()

        assert normalizer.f0_log_transform is True
        assert normalizer.f0_mean == 150.0
        assert normalizer.f0_std == 50.0

    def test_normalization_denormalization(self, sample_features):
        """Test normalization and denormalization round trip."""
        normalizer = ProsodyNormalizer()

        # Normalize
        features_norm = normalizer(sample_features)

        # Denormalize
        features_denorm = normalizer.denormalize(features_norm)

        # Should be close to original (excluding numerical precision)
        # Only check voiced frames for F0
        voiced_mask = sample_features[:, :, 3] > 0.5

        if voiced_mask.any():
            f0_orig = sample_features[:, :, 0][voiced_mask]
            f0_denorm = features_denorm[:, :, 0][voiced_mask]

            # Allow small numerical differences
            assert torch.allclose(f0_orig, f0_denorm, rtol=1e-3, atol=1.0)

        # Energy should be exact (linear transformation)
        energy_orig = sample_features[:, :, 1]
        energy_denorm = features_denorm[:, :, 1]
        assert torch.allclose(energy_orig, energy_denorm, rtol=1e-5)

    def test_f0_log_normalization(self, sample_features):
        """Test F0 log normalization specifically."""
        normalizer = ProsodyNormalizer(f0_log_transform=True)

        features_norm = normalizer(sample_features)

        # Check that unvoiced frames remain 0
        unvoiced_mask = sample_features[:, :, 0] == 0
        assert torch.all(features_norm[:, :, 0][unvoiced_mask] == 0)

        # Check that voiced frames are normalized
        voiced_mask = sample_features[:, :, 0] > 0
        if voiced_mask.any():
            f0_norm_voiced = features_norm[:, :, 0][voiced_mask]

            # Should have different scale than original
            f0_orig_voiced = sample_features[:, :, 0][voiced_mask]
            assert not torch.allclose(f0_norm_voiced, f0_orig_voiced)

    def test_linear_f0_normalization(self, sample_features):
        """Test linear F0 normalization."""
        normalizer = ProsodyNormalizer(f0_log_transform=False)

        features_norm = normalizer(sample_features)

        # Check normalization of voiced frames
        voiced_mask = sample_features[:, :, 0] > 0
        if voiced_mask.any():
            f0_orig = sample_features[:, :, 0][voiced_mask]
            f0_norm = features_norm[:, :, 0][voiced_mask]

            # Should be normalized around 0
            f0_norm_mean = f0_norm.mean()
            assert abs(f0_norm_mean) < 1.0  # Should be close to 0

    def test_energy_normalization(self, sample_features):
        """Test energy normalization."""
        normalizer = ProsodyNormalizer()

        features_norm = normalizer(sample_features)

        # Energy should be normalized
        energy_norm = features_norm[:, :, 1]
        energy_norm_mean = energy_norm.mean()
        energy_norm_std = energy_norm.std()

        # Should be approximately standard normal
        assert abs(energy_norm_mean) < 0.5
        assert abs(energy_norm_std - 1.0) < 0.5

    def test_vad_voicing_unchanged(self, sample_features):
        """Test that VAD and voicing are unchanged by normalization."""
        normalizer = ProsodyNormalizer()

        features_norm = normalizer(sample_features)

        # VAD and voicing should be unchanged
        assert torch.allclose(features_norm[:, :, 2], sample_features[:, :, 2])
        assert torch.allclose(features_norm[:, :, 3], sample_features[:, :, 3])


class TestProsodyValidation:
    """Test prosody feature validation."""

    def test_valid_features(self):
        """Test validation with valid features."""
        batch_size, seq_len = 2, 20
        features = torch.zeros(batch_size, seq_len, 4)

        # Set realistic values
        features[:, :, 0] = (
            torch.linspace(100, 200, seq_len).unsqueeze(0).repeat(batch_size, 1)
        )  # F0
        features[:, :, 1] = (
            torch.linspace(-8, -3, seq_len).unsqueeze(0).repeat(batch_size, 1)
        )  # Energy
        features[:, :, 2] = torch.ones(batch_size, seq_len) * 0.8  # VAD
        features[:, :, 3] = torch.ones(batch_size, seq_len) * 0.9  # Voicing

        results = validate_prosody_features(features)

        assert results["valid"] is True
        assert "f0_mean" in results["stats"]
        assert "vad_rate" in results["stats"]

        # Check reasonable F0 range
        assert 80 < results["stats"]["f0_mean"] < 300

    def test_invalid_shape(self):
        """Test validation with invalid shape."""
        invalid_features = torch.randn(2, 20, 3)  # Wrong last dimension

        results = validate_prosody_features(invalid_features)

        assert results["valid"] is False
        assert any("Expected shape" in w for w in results["warnings"])

    def test_no_voiced_frames(self):
        """Test validation with no voiced frames."""
        batch_size, seq_len = 1, 10
        features = torch.zeros(batch_size, seq_len, 4)
        # F0 remains 0 (unvoiced), other features can be non-zero
        features[:, :, 1] = -5.0  # Energy
        features[:, :, 2] = 0.1  # Low VAD
        features[:, :, 3] = 0.1  # Low voicing

        results = validate_prosody_features(features)

        assert any("No voiced frames" in w for w in results["warnings"])

    def test_unusual_f0_range(self):
        """Test validation with unusual F0 range."""
        batch_size, seq_len = 1, 10
        features = torch.zeros(batch_size, seq_len, 4)

        # Set unrealistic F0 values
        features[:, :, 0] = 30.0  # Too low
        features[:, :, 3] = 0.9  # High voicing to make it count as voiced

        results = validate_prosody_features(features)

        assert any("F0 range unusual" in w for w in results["warnings"])

    def test_extreme_vad_rates(self):
        """Test validation with extreme VAD rates."""
        batch_size, seq_len = 1, 20

        # Test very low VAD rate
        features_low = torch.zeros(batch_size, seq_len, 4)
        features_low[:, :, 2] = 0.05  # Very low VAD

        results_low = validate_prosody_features(features_low)
        assert any("Very low VAD rate" in w for w in results_low["warnings"])

        # Test very high VAD rate
        features_high = torch.zeros(batch_size, seq_len, 4)
        features_high[:, :, 2] = 0.95  # Very high VAD

        results_high = validate_prosody_features(features_high)
        assert any("Very high VAD rate" in w for w in results_high["warnings"])

    def test_nan_detection(self):
        """Test detection of NaN values."""
        batch_size, seq_len = 1, 10
        features = torch.zeros(batch_size, seq_len, 4)
        features[0, 0, 0] = float("nan")  # Insert NaN

        results = validate_prosody_features(features)

        assert results["valid"] is False
        assert any("NaN values" in w for w in results["warnings"])

    def test_inf_detection(self):
        """Test detection of infinite values."""
        batch_size, seq_len = 1, 10
        features = torch.zeros(batch_size, seq_len, 4)
        features[0, 0, 1] = float("inf")  # Insert infinity

        results = validate_prosody_features(features)

        assert results["valid"] is False
        assert any("Infinite values" in w for w in results["warnings"])

    def test_statistics_computation(self):
        """Test that statistics are computed correctly."""
        batch_size, seq_len = 2, 10
        features = torch.zeros(batch_size, seq_len, 4)

        # Set known values
        f0_values = torch.tensor([100.0, 150.0, 200.0])
        features[0, :3, 0] = f0_values  # First 3 frames voiced
        features[:, :, 3] = torch.where(
            torch.arange(seq_len).unsqueeze(0) < 3, torch.tensor(0.9), torch.tensor(0.1)
        )  # Voicing matches F0

        results = validate_prosody_features(features)

        # Check F0 statistics
        expected_f0_mean = f0_values.mean().item()
        assert abs(results["stats"]["f0_mean"] - expected_f0_mean) < 1e-3
