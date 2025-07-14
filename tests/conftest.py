"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rate():
    """Standard sample rate for testing."""
    return 16000


@pytest.fixture
def device():
    """Device for testing (prefer CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_audio(sample_rate):
    """Generate dummy audio for testing."""
    duration = 2.0  # seconds
    samples = int(sample_rate * duration)
    # Generate sine wave with some noise
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(samples)
    return torch.from_numpy(audio)


@pytest.fixture
def dummy_arkit_blendshapes():
    """Generate dummy ARKit 52 blendshapes."""
    # ARKit blendshapes are typically in [0, 1] range
    return torch.rand(52, dtype=torch.float32)


@pytest.fixture
def dummy_batch_audio(dummy_audio):
    """Generate batch of dummy audio."""
    return dummy_audio.unsqueeze(0).repeat(4, 1)  # Batch size 4


@pytest.fixture
def dummy_batch_blendshapes(dummy_arkit_blendshapes):
    """Generate batch of dummy blendshapes."""
    return dummy_arkit_blendshapes.unsqueeze(0).repeat(4, 1)  # Batch size 4


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir