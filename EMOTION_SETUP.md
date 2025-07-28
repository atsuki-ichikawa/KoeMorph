# KoeMorph Dual-Stream Emotion Processing Setup

This guide helps you set up the enhanced dual-stream emotion processing system for KoeMorph.

## 🎯 Overview

The new system provides:
- **Robust emotion2vec integration** with automatic fallback
- **Stream specialization**: mel-spectrograms → mouth movements, emotions → facial expressions  
- **Multi-backend support**: emotion2vec, OpenSMILE, basic prosodic features
- **Comprehensive monitoring** and debugging tools

## 📦 Installation

### 1. Core Dependencies
```bash
# Install KoeMorph with emotion support
pip install -e .[emotion2vec]

# Or install manually
pip install funasr modelscope
pip install opensmile  # Optional fallback
```

### 2. Model Downloads
The emotion2vec model will download automatically on first use (~300MB).

### 3. Verify Installation
```bash
python test_emotion_processing.py
```

## 🚀 Quick Start

### Basic Training
```bash
# Train with emotion2vec (will fallback to OpenSMILE if needed)
python src/train_dual_stream.py --config-name dual_stream_config

# Train with OpenSMILE backend (faster)
python src/train_dual_stream.py --config-name dual_stream_config \
    model.emotion_config.backend=opensmile

# Train with basic features (fastest)
python src/train_dual_stream.py --config-name dual_stream_config \
    model.emotion_config.backend=basic
```

### Model Variants
```bash
# High-quality model
python src/train_dual_stream.py --config-name dual_stream_config \
    model=dual_stream/variants/high_quality

# Lightweight model  
python src/train_dual_stream.py --config-name dual_stream_config \
    model=dual_stream/variants/lightweight

# No emotion2vec (OpenSMILE fallback)
python src/train_dual_stream.py --config-name dual_stream_config \
    model=dual_stream/variants/no_emotion2vec
```

## ⚙️ Configuration

### Emotion Backend Options

#### emotion2vec (Recommended)
```yaml
emotion_config:
  backend: "emotion2vec"
  model_name: "iic/emotion2vec_plus_large"
  device: "auto"
  enable_caching: true
  batch_size: 4
```

#### OpenSMILE (Fast Fallback)
```yaml
emotion_config:
  backend: "opensmile"
  enable_caching: true
  batch_size: 8  # Can handle larger batches
```

#### Basic Features (Minimal)
```yaml
emotion_config:
  backend: "basic"
  enable_caching: false
```

### Stream Specialization
```yaml
dual_stream_attention:
  specialization_factor: 3.0  # Higher = stronger separation
  use_learnable_weights: true
  temperature: 1.0
```

## 📊 Monitoring

### Enable Monitoring
```yaml
monitoring:
  enable: true
  log_dir: "logs/emotion_monitor"
  enable_plotting: true
  verbose: false
```

### View Results
```python
from src.utils.emotion_monitor import get_monitor

monitor = get_monitor()
stats = monitor.get_statistics()
report = monitor.generate_report()
monitor.plot_performance_metrics()
```

## 🔧 Troubleshooting

### emotion2vec Not Working
The system automatically falls back to OpenSMILE or basic features:

1. **Check installation**: `pip install funasr modelscope`
2. **Verify model download**: First run downloads ~300MB
3. **Check logs**: Look for "Fallback from emotion2vec to..." messages
4. **Use alternative**: Set `backend: "opensmile"` in config

### Memory Issues
```yaml
# Reduce memory usage
emotion_config:
  batch_size: 1  # Smaller batches
  enable_caching: false  # Disable caching
model:
  d_model: 128  # Smaller model
  mel_sequence_length: 128
```

### Performance Issues  
```yaml
# Optimize for speed
emotion_config:
  backend: "opensmile"  # Fastest reliable option
  batch_size: 8
temporal_smoothing:
  enable: false  # Skip smoothing for speed
```

## 📈 Features

### Stream Differentiation
- **Mel-stream**: Frequency bands (0-80) → mouth blendshapes (jaw, lips)
- **Emotion-stream**: Emotion features → expression blendshapes (eyes, brows, cheeks)

### Automatic Fallback Chain
1. **emotion2vec** (best quality, slower)
2. **OpenSMILE** (good quality, faster) 
3. **Basic prosodic** (minimal quality, fastest)

### Blendshape Mapping
- Emotions automatically mapped to appropriate blendshapes
- `angry` → brow down, eye squint, nose sneer
- `happy` → cheek squint, eye squint, brow up
- `sad` → brow inner up, eye squint
- etc.

## 🎛️ Advanced Usage

### Custom Emotion Mapping
```python
from src.features.emotion_extractor import EMOTION_TO_BLENDSHAPE_MAPPING

# Modify emotion mappings
EMOTION_TO_BLENDSHAPE_MAPPING["custom_emotion"] = {
    "browDownLeft": 0.8,
    "browDownRight": 0.8,
    # ... more mappings
}
```

### Monitoring Integration
```python
# In training loop
from src.utils.emotion_monitor import get_monitor

monitor = get_monitor()
# Automatic logging happens in EmotionExtractor
# Access stats anytime:
stats = monitor.get_statistics()
```

## 🆘 Support

- **Test script**: `python test_emotion_processing.py`
- **Monitor logs**: Check `logs/emotion_monitor/`
- **Model info**: Call `model.get_model_info()`
- **Debug**: Set `monitoring.verbose: true` in config