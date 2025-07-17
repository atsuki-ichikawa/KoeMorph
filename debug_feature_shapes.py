#!/usr/bin/env python3
"""Debug feature extractor output shapes."""

import torch
from src.data.dataset import KoeMorphDataModule
from src.features.stft import MelSpectrogramExtractor
from src.features.prosody import ProsodyExtractor
from src.features.emotion2vec import Emotion2VecExtractor

def debug_feature_shapes():
    """Debug the shapes of different feature extractors."""
    
    # Setup data
    data_module = KoeMorphDataModule(
        train_data_dir="/home/ichikawa/KoeMorph/data/train",
        val_data_dir=None,
        test_data_dir=None,
        sample_rate=16000,
        max_audio_length=10.0,
        target_fps=30,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    
    data_module.setup()
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    audio = batch['wav']
    print(f"Input audio shape: {audio.shape}")
    print(f"Input audio length: {audio.shape[1]} samples ({audio.shape[1] / 16000:.3f} seconds)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create feature extractors
    mel_extractor = MelSpectrogramExtractor(
        sample_rate=16000,
        target_fps=30,
        n_fft=512,
        n_mels=80,
        f_min=80,
        f_max=8000,
    ).to(device)
    
    prosody_extractor = ProsodyExtractor(
        sample_rate=16000,
        target_fps=30,
        frame_length=0.025,
        frame_shift=0.01,
    ).to(device)
    
    emotion_extractor = Emotion2VecExtractor(
        model_name="emotion2vec_base",
        target_fps=30,
        sample_rate=16000,
        freeze_pretrained=True,
        output_dim=256,
    ).to(device)
    
    # Move audio to device
    audio = audio.to(device)
    
    # Extract features
    print("\n=== Feature Extraction ===")
    
    # Mel spectrogram
    mel_features = mel_extractor(audio)
    print(f"Mel features shape: {mel_features.shape}")
    print(f"Mel frames: {mel_features.shape[1]}")
    
    # Prosody
    prosody_features = prosody_extractor(audio)
    print(f"Prosody features shape: {prosody_features.shape}")
    print(f"Prosody frames: {prosody_features.shape[1]}")
    
    # Emotion2vec
    emotion_features = emotion_extractor(audio)
    print(f"Emotion features shape: {emotion_features.shape}")
    print(f"Emotion frames: {emotion_features.shape[1]}")
    
    # Check time alignment
    print(f"\n=== Time Alignment ===")
    expected_frames = int(audio.shape[1] / 16000 * 30)  # Expected frames at 30 FPS
    print(f"Expected frames at 30 FPS: {expected_frames}")
    
    # Check differences
    mel_diff = mel_features.shape[1] - expected_frames
    prosody_diff = prosody_features.shape[1] - expected_frames
    emotion_diff = emotion_features.shape[1] - expected_frames
    
    print(f"Mel difference: {mel_diff}")
    print(f"Prosody difference: {prosody_diff}")
    print(f"Emotion difference: {emotion_diff}")
    
    # Check if they match each other
    print(f"\n=== Feature Alignment ===")
    mel_vs_prosody = mel_features.shape[1] - prosody_features.shape[1]
    mel_vs_emotion = mel_features.shape[1] - emotion_features.shape[1]
    prosody_vs_emotion = prosody_features.shape[1] - emotion_features.shape[1]
    
    print(f"Mel vs Prosody: {mel_vs_prosody}")
    print(f"Mel vs Emotion: {mel_vs_emotion}")
    print(f"Prosody vs Emotion: {prosody_vs_emotion}")
    
    if mel_vs_prosody == 0 and mel_vs_emotion == 0 and prosody_vs_emotion == 0:
        print("✅ All features are aligned!")
    else:
        print("❌ Features are not aligned!")
        print("Need to fix time alignment in feature extractors")

if __name__ == "__main__":
    debug_feature_shapes()