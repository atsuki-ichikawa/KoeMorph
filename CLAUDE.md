# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Environment Setup
```bash
# Install core dependencies
pip install -e .

# Install development dependencies (includes testing, linting)
pip install -e .[dev]

# Install real-time inference dependencies
pip install -e .[realtime]

# Install emotion2vec support
pip install -e .[emotion2vec]
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/model/test_attention.py -v

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code with ruff
ruff check src/ tests/

# Type checking (if mypy is configured)
# No mypy configuration found in project
```

### Training
```bash
# Train with default configuration
python src/train.py

# Train with custom config
python src/train.py --config-path configs --config-name custom_config

# Debug mode (limited batches)
python src/train.py debug=true
```

### Real-time Inference
```bash
# Real-time inference with UDP output
python scripts/rt.py --model_path checkpoints/best_model.pth --output_mode udp

# With OSC output for Unity/Unreal
python scripts/rt.py --model_path checkpoints/best_model.pth --output_mode osc --port 9001
```

### Model Export
```bash
# Export to TorchScript and ONNX
python scripts/export_model.py --model_path checkpoints/best_model.pth --formats torchscript onnx

# Mobile-optimized export
python scripts/export_model.py --model_path checkpoints/best_model.pth --formats torchscript --mobile_optimize
```

## Architecture Overview

**KoeMorph** is a real-time facial expression generation system that converts audio to ARKit 52 blendshapes using cross-attention architecture.

### Key Components

**Multi-Stream Audio Processing:**
- `src/features/stft.py`: Mel-spectrogram extraction (30 FPS)
- `src/features/prosody.py`: F0, energy, VAD features
- `src/features/emotion2vec.py`: Emotion embeddings from pretrained models

**Core Model Architecture (`src/model/`):**
- `gaussian_face.py`: Main KoeMorphModel class combining all components
- `attention.py`: Cross-attention modules with ARKit blendshapes as queries, audio features as keys/values
- `decoder.py`: Blendshape decoder with temporal smoothing and constraints
- `losses.py`: Multi-component loss functions and evaluation metrics

**Data Pipeline (`src/data/`):**
- `io.py`: ARKit JSONL + WAV file loading
- `dataset.py`: PyTorch Dataset/DataLoader with feature synchronization

### Configuration System

Uses Hydra for configuration management:
- `configs/config.yaml`: Main configuration
- `configs/data/default.yaml`: Data loading settings
- `configs/model/default.yaml`: Model architecture parameters
- `configs/training/default.yaml`: Training hyperparameters

### Training Pipeline

The `KoeMorphTrainer` class in `src/train.py` handles:
- Multi-stream feature extraction during training
- Cross-attention training with blendshape targets
- Validation and checkpointing
- TensorBoard logging

### Real-time Architecture

**Causal Design:** Model supports causal attention with configurable window size for real-time inference.

**Temporal Processing:** Built-in temporal smoothing and blendshape constraints ensure realistic output.

**Feature Extraction:** All audio features can be computed in real-time at 30 FPS.

## Data Format

**Audio Input:** 16kHz WAV files, mono preferred

**ARKit Blendshapes:** JSONL format with synchronized timestamps:
```json
{"timestamp": 0.033, "blendshapes": [0.0, 0.2, 0.8, ...]}
```

## Important Notes

- Model outputs 52 ARKit blendshape coefficients in [0,1] range
- Target frame rate: 30 FPS for real-time applications
- Uses PyTorch 2.0+ with optional TorchScript export for mobile deployment
- Project recently renamed from "GaussianFace" to "KoeMorph"