"""Tests for data.io module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.io import ARKitDataLoader, validate_data_consistency


class TestARKitDataLoader:
    """Test ARKit data loading functionality."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample test data files."""
        # Create sample audio (2 seconds at 16kHz)
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        audio = 0.1 * np.random.randn(samples).astype(np.float32)

        wav_path = tmp_path / "test_audio.wav"
        sf.write(str(wav_path), audio, sample_rate)

        # Create sample blendshapes (60 frames at 30 FPS = 2 seconds)
        num_frames = 60
        jsonl_path = tmp_path / "test_blendshapes.jsonl"

        with open(jsonl_path, "w") as f:
            for i in range(num_frames):
                timestamp = i / 30.0  # 30 FPS
                blendshapes = np.random.rand(52).tolist()  # Random values [0,1]
                data = {"timestamp": timestamp, "blendshapes": blendshapes}
                f.write(json.dumps(data) + "\n")

        return wav_path, jsonl_path

    def test_load_sample_basic(self, sample_data):
        """Test basic sample loading."""
        wav_path, jsonl_path = sample_data
        loader = ARKitDataLoader()

        sample = loader.load_sample(jsonl_path, wav_path)

        assert "wav" in sample
        assert "arkit" in sample
        assert isinstance(sample["wav"], torch.Tensor)
        assert isinstance(sample["arkit"], torch.Tensor)

        # Check shapes
        assert sample["wav"].dim() == 1  # (L,)
        assert sample["arkit"].shape == (60, 52)  # (T, 52)

    def test_load_sample_file_not_found(self):
        """Test error handling for missing files."""
        loader = ARKitDataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_sample("nonexistent.jsonl", "nonexistent.wav")

    def test_blendshape_count_validation(self, tmp_path):
        """Test validation of blendshape count."""
        # Create invalid blendshape file (wrong number of values)
        jsonl_path = tmp_path / "invalid_blendshapes.jsonl"
        with open(jsonl_path, "w") as f:
            data = {"timestamp": 0.0, "blendshapes": [0.5] * 51}  # Wrong count!
            f.write(json.dumps(data) + "\n")

        # Create dummy audio
        wav_path = tmp_path / "dummy.wav"
        sf.write(str(wav_path), np.random.randn(1000), 16000)

        loader = ARKitDataLoader()
        with pytest.raises(ValueError, match="Expected 52 blendshapes"):
            loader.load_sample(jsonl_path, wav_path)

    def test_time_sync_validation(self, tmp_path):
        """Test temporal synchronization validation."""
        # Create audio file (1 second)
        wav_path = tmp_path / "short_audio.wav"
        sf.write(str(wav_path), np.random.randn(16000), 16000)

        # Create blendshape file (2 seconds worth)
        jsonl_path = tmp_path / "long_blendshapes.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(60):  # 2 seconds at 30 FPS
                data = {"timestamp": i / 30.0, "blendshapes": [0.5] * 52}
                f.write(json.dumps(data) + "\n")

        loader = ARKitDataLoader(max_time_drift=0.1)
        with pytest.raises(ValueError, match="Time drift"):
            loader.load_sample(jsonl_path, wav_path)

    def test_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON."""
        jsonl_path = tmp_path / "invalid.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("not valid json\n")

        wav_path = tmp_path / "dummy.wav"
        sf.write(str(wav_path), np.random.randn(1000), 16000)

        loader = ARKitDataLoader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_sample(jsonl_path, wav_path)

    def test_missing_fields(self, tmp_path):
        """Test error handling for missing required fields."""
        jsonl_path = tmp_path / "missing_fields.jsonl"
        with open(jsonl_path, "w") as f:
            # Missing 'blendshapes' field
            data = {"timestamp": 0.0}
            f.write(json.dumps(data) + "\n")

        wav_path = tmp_path / "dummy.wav"
        sf.write(str(wav_path), np.random.randn(1000), 16000)

        loader = ARKitDataLoader()
        with pytest.raises(ValueError, match="Missing 'blendshapes' field"):
            loader.load_sample(jsonl_path, wav_path)

    def test_mono_conversion(self, tmp_path):
        """Test stereo to mono conversion."""
        # Create stereo audio
        stereo_audio = np.random.randn(1000, 2).astype(np.float32)
        wav_path = tmp_path / "stereo.wav"
        sf.write(str(wav_path), stereo_audio, 16000)

        # Create minimal blendshape file
        jsonl_path = tmp_path / "minimal.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(2):  # Very short
                data = {"timestamp": i * 0.5, "blendshapes": [0.5] * 52}
                f.write(json.dumps(data) + "\n")

        loader = ARKitDataLoader(max_time_drift=0.5)
        sample = loader.load_sample(jsonl_path, wav_path)

        # Should be converted to mono
        assert sample["wav"].dim() == 1
        assert len(sample["wav"]) == 1000

    def test_load_batch(self, tmp_path):
        """Test batch loading functionality."""
        # Create multiple sample pairs
        pairs = []
        for i in range(3):
            # Audio
            wav_path = tmp_path / f"audio_{i}.wav"
            audio = 0.1 * np.random.randn(8000).astype(np.float32)  # 0.5 seconds
            sf.write(str(wav_path), audio, 16000)

            # Blendshapes
            jsonl_path = tmp_path / f"blendshapes_{i}.jsonl"
            with open(jsonl_path, "w") as f:
                for j in range(15):  # 0.5 seconds at 30 FPS
                    data = {
                        "timestamp": j / 30.0,
                        "blendshapes": np.random.rand(52).tolist(),
                    }
                    f.write(json.dumps(data) + "\n")

            pairs.append((jsonl_path, wav_path))

        loader = ARKitDataLoader()
        samples = loader.load_batch(pairs)

        assert len(samples) == 3
        for sample in samples:
            assert "wav" in sample
            assert "arkit" in sample
            assert sample["arkit"].shape[1] == 52


class TestDataValidation:
    """Test data validation utilities."""

    def test_validate_data_consistency(self):
        """Test data consistency validation."""
        # Create consistent samples
        samples = []
        for i in range(3):
            sample = {
                "wav": torch.randn(16000),  # 1 second at 16kHz
                "arkit": torch.rand(30, 52),  # 1 second at 30 FPS
            }
            samples.append(sample)

        results = validate_data_consistency(samples)
        assert results["valid"] is True

    def test_validate_inconsistent_blendshapes(self):
        """Test detection of inconsistent blendshape dimensions."""
        samples = [
            {"wav": torch.randn(8000), "arkit": torch.rand(15, 52)},  # Correct
            {"wav": torch.randn(8000), "arkit": torch.rand(15, 51)},  # Wrong dimension!
        ]

        results = validate_data_consistency(samples)
        assert results["valid"] is False
        assert any("blendshape dimensions" in issue for issue in results["issues"])

    def test_validate_duration_mismatch(self):
        """Test detection of audio/blendshape duration mismatches."""
        samples = [
            {
                "wav": torch.randn(16000),  # 1 second
                "arkit": torch.rand(60, 52),  # 2 seconds at 30 FPS
            }
        ]

        results = validate_data_consistency(samples, tolerance=0.05)
        assert results["valid"] is False
        assert any("duration mismatch" in issue for issue in results["issues"])

    def test_validate_empty_samples(self):
        """Test validation with empty sample list."""
        results = validate_data_consistency([])
        assert results["valid"] is False
        assert results["reason"] == "No samples provided"
