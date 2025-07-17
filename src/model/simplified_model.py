"""
Simplified KoeMorph model for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class SimplifiedKoeMorphModel(nn.Module):
    """
    Simplified KoeMorph model that processes audio directly.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_query: int = 256,
        d_key: int = 256,
        d_value: int = 256,
        audio_encoder: dict = None,
        attention: dict = None,
        decoder: dict = None,
        smoothing: dict = None,
        num_blendshapes: int = 52,
        sample_rate: int = 16000,
        target_fps: int = 30,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        
        # Audio processing parameters
        self.n_mels = 80
        self.hop_length = int(sample_rate // target_fps)  # 533 for 16kHz, 30fps
        self.n_fft = 1024
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.n_mels, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Attention mechanism (simplified)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention.get('num_heads', 8) if attention else 8,
            dropout=attention.get('dropout', 0.1) if attention else 0.1,
            batch_first=True
        )
        
        # Blendshape decoder
        decoder_hidden = decoder.get('hidden_dim', 128) if decoder else 128
        self.decoder = nn.Sequential(
            nn.Linear(d_model, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, num_blendshapes),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )
        
        # Learnable blendshape queries
        self.blendshape_queries = nn.Parameter(
            torch.randn(num_blendshapes, d_model) * 0.1
        )
        
    def extract_mel_features(self, audio):
        """Extract mel-spectrogram features from audio."""
        batch_size, audio_length = audio.shape
        
        # Convert to numpy for librosa processing
        mel_features = []
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            
            # Extract mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=80,
                fmax=8000
            )
            
            # Convert to log scale
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = (mel + 80) / 80  # Normalize to [0, 1]
            
            mel_features.append(mel.T)  # (T, n_mels)
        
        # Find max length and pad
        max_length = max(feat.shape[0] for feat in mel_features)
        padded_features = np.zeros((batch_size, max_length, self.n_mels))
        
        for i, feat in enumerate(mel_features):
            padded_features[i, :feat.shape[0], :] = feat
        
        return torch.tensor(padded_features, dtype=torch.float32, device=audio.device)
    
    def forward(self, audio):
        """
        Forward pass.
        
        Args:
            audio: Audio tensor of shape (B, T)
            
        Returns:
            Blendshape predictions of shape (B, 52)
        """
        batch_size = audio.shape[0]
        
        # Extract mel features
        mel_features = self.extract_mel_features(audio)  # (B, T, n_mels)
        
        # Encode audio features
        audio_encoded = self.audio_encoder(mel_features)  # (B, T, d_model)
        
        # Prepare queries (repeat for each batch)
        queries = self.blendshape_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 52, d_model)
        
        # Apply attention
        attn_output, _ = self.attention(
            query=queries,
            key=audio_encoded,
            value=audio_encoded,
            need_weights=False
        )  # (B, 52, d_model)
        
        # Decode to blendshapes
        blendshapes = self.decoder(attn_output)  # (B, 52, num_blendshapes)
        
        # Average over blendshape dimension to get final output
        blendshapes = blendshapes.mean(dim=1)  # (B, num_blendshapes)
        
        return blendshapes
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_temporal_state(self):
        """Reset temporal state for new sequence (no-op for simplified model)."""
        pass