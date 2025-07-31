# New KoeMorph Data Format (v2.0) Training Guide

## Overview

The KoeMorph project now supports the new v2.0 data format with the following features:
- Folder-based structure with metadata files
- SMPTE timecode synchronization
- 60fps support (automatically detected)
- Automatic FPS resampling when needed

## Using the New Data Format

### 1. Data Validation

First, validate your data using the validation script:

```bash
# Validate all data
python validate_koemorph_data.py /home/ichikawa/KoeMorph/koemorph_learning_data/data/processed --fps 60

# Save detailed results
python validate_koemorph_data.py /home/ichikawa/KoeMorph/koemorph_learning_data/data/processed --fps 60 --output validation_results.json
```

### 2. Training with New Data

To train with the new data format, use the `koemorph_v2` data configuration:

```bash
# Train with new data format
python src/train_sequential.py data=koemorph_v2

# Customize training parameters
python src/train_sequential.py data=koemorph_v2 training.learning_rate=5e-5 training.max_epochs=100

# Use specific window/stride settings
python src/train_sequential.py data=koemorph_v2 data.sequential.window_frames=512 data.sequential.stride_frames=128
```

### 3. Configuration Details

The new data configuration (`configs/data/koemorph_v2.yaml`) includes:

- **Data paths**: Points to `/home/ichikawa/KoeMorph/koemorph_learning_data/data/processed`
- **Automatic splitting**: Since all data is in one directory, it automatically splits into train/val/test (80/10/10)
- **FPS support**: Automatically detects source FPS from metadata and resamples if needed
- **Sequential settings**: 
  - `window_frames`: 512 frames (~8.5 seconds at 60fps)
  - `stride_frames`: 256 frames (50% overlap)

### 4. Key Differences from Old Format

| Feature | Old Format | New Format (v2.0) |
|---------|------------|-------------------|
| Structure | Individual audio/JSONL files | Folder-based with metadata |
| FPS | Fixed 30fps | Auto-detected (60fps default) |
| Synchronization | Frame-based | SMPTE timecode-based |
| Metadata | None | metadata.json per recording |
| Validation | Manual | Built-in validation script |

### 5. Training Tips

1. **FPS Handling**: The model automatically adapts to the data's FPS. No manual configuration needed.

2. **Memory Usage**: 60fps data uses twice the memory. Adjust batch size if needed:
   ```bash
   python src/train_sequential.py data=koemorph_v2 data.batch_size=8
   ```

3. **Window Size**: For 60fps data, you might want larger windows:
   ```bash
   python src/train_sequential.py data=koemorph_v2 data.sequential.window_frames=768
   ```

4. **Monitoring**: Check TensorBoard for training progress:
   ```bash
   tensorboard --logdir outputs/
   ```

### 6. Compatibility

The updated `train_sequential.py` supports both old and new data formats:
- It automatically detects the data format from the configuration
- No code changes needed when switching between formats
- Models trained with 30fps can be fine-tuned with 60fps data

### 7. Example Training Session

```bash
# Full training pipeline
cd /home/ichikawa/KoeMorph

# 1. Validate data
python validate_koemorph_data.py koemorph_learning_data/data/processed

# 2. Start training
python src/train_sequential.py data=koemorph_v2 \
    training.max_epochs=50 \
    training.learning_rate=1e-4 \
    training.checkpoint_frequency=5

# 3. Monitor progress
tensorboard --logdir outputs/
```

## Troubleshooting

- **Out of Memory**: Reduce batch size or window frames
- **Slow Training**: Reduce stride_frames for less dense sampling
- **FPS Mismatch Warnings**: Normal if data has mixed FPS; automatic resampling handles it