"""
Dual-stream cross-attention module for independent log-Mel and emotion2vec processing.

This module processes log-Mel features (for mouth movements) and emotion2vec features
(for facial expressions) independently, allowing for more interpretable attention patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ARKit blendshape grouping
MOUTH_BLENDSHAPES = [
    # Jaw movements
    'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
    # Mouth shapes
    'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
    # Tongue (if visible affects mouth shape)
    'tongueOut'
]

# ARKit blendshape indices (0-indexed)
ARKIT_BLENDSHAPES = [
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft',
    'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight',
    'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight', 'jawForward',
    'jawLeft', 'jawRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker',
    'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft',
    'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
    'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut'
]

# Get mouth blendshape indices
MOUTH_INDICES = [i for i, name in enumerate(ARKIT_BLENDSHAPES) if name in MOUTH_BLENDSHAPES]
EXPRESSION_INDICES = [i for i in range(52) if i not in MOUTH_INDICES]


class DualStreamCrossAttention(nn.Module):
    """
    Dual-stream cross-attention for separate processing of mel-spectrogram and emotion features.
    
    This module implements two independent attention mechanisms:
    1. Mel-attention: Focuses on mouth-related blendshapes using frequency-domain features
    2. Emotion-attention: Focuses on expression-related blendshapes using emotion embeddings
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_mel_channels: int = 80,
        mel_sequence_length: int = 256,
        mel_temporal_frames: int = 3,  # Additional temporal frames for mouth detail
        emotion_dim: int = 256,  # Concatenated + compressed dimension (3×88→256)
        emotion_sequence_length: int = 1,   # Single compressed vector
        dropout: float = 0.1,
        num_blendshapes: int = 52,
        use_learnable_weights: bool = True,
        temperature: float = 1.0,
    ):
        """
        Initialize dual-stream cross-attention module.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_mel_channels: Number of mel frequency channels (80)
            mel_sequence_length: Sequence length for each mel channel (256)
            mel_temporal_frames: Additional temporal frames for mouth detail (3)
            emotion_dim: Dimension of concatenated eGeMAPS features (256)
            emotion_sequence_length: Sequence length (1 for concatenated approach)
            dropout: Dropout rate
            num_blendshapes: Number of ARKit blendshapes (52)
            use_learnable_weights: Whether to use learnable stream weights
            temperature: Temperature for attention scaling
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_mel_channels = num_mel_channels
        self.mel_sequence_length = mel_sequence_length
        self.mel_temporal_frames = mel_temporal_frames
        self.emotion_dim = emotion_dim
        self.emotion_sequence_length = emotion_sequence_length
        self.num_blendshapes = num_blendshapes
        self.temperature = temperature
        
        # Enhanced mel-spectrogram processing for mouth movements
        # Long-term context: 80 × 256 = 20,480 + Short-term detail: 80 × 3 = 240 = 20,720 total
        self.total_mel_dim = num_mel_channels * (mel_sequence_length + mel_temporal_frames)
        self.mel_channel_encoder = nn.Linear(mel_sequence_length + mel_temporal_frames, d_model)
        self.mel_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Concatenated eGeMAPS processing for facial expressions
        self.emotion_encoder = nn.Linear(emotion_dim, d_model)
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Blendshape-specific query embeddings
        self.mouth_queries = nn.Parameter(torch.randn(len(MOUTH_INDICES), d_model) * 0.02)
        self.expression_queries = nn.Parameter(torch.randn(len(EXPRESSION_INDICES), d_model) * 0.02)
        
        # Learnable stream weights for soft assignment
        if use_learnable_weights:
            # Initialize with bias towards intended streams
            self.mel_weights = nn.Parameter(torch.ones(num_blendshapes))
            self.emotion_weights = nn.Parameter(torch.ones(num_blendshapes))
            
            # Initialize with stronger weights for intended blendshapes
            with torch.no_grad():
                self.mel_weights[MOUTH_INDICES] = 2.0
                self.mel_weights[EXPRESSION_INDICES] = 0.5
                self.emotion_weights[MOUTH_INDICES] = 0.5
                self.emotion_weights[EXPRESSION_INDICES] = 2.0
        else:
            # Fixed weights
            mel_weights = torch.zeros(num_blendshapes)
            emotion_weights = torch.zeros(num_blendshapes)
            mel_weights[MOUTH_INDICES] = 1.0
            emotion_weights[EXPRESSION_INDICES] = 1.0
            
            self.register_buffer('mel_weights', mel_weights)
            self.register_buffer('emotion_weights', emotion_weights)
        
        # Output projections
        self.mel_output_proj = nn.Linear(d_model, d_model)
        self.emotion_output_proj = nn.Linear(d_model, d_model)
        
        # Final blendshape decoder
        self.blendshape_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.mel_norm = nn.LayerNorm(d_model)
        self.emotion_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        mel_features: torch.Tensor,
        mel_temporal_features: torch.Tensor,
        emotion_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of enhanced dual-stream cross-attention.
        
        Args:
            mel_features: Long-term mel-spectrogram features of shape (B, T, 80) 
            mel_temporal_features: Short-term mel features of shape (B, 3, 80)
            emotion_features: Concatenated eGeMAPS features of shape (B, 256)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - blendshapes: Final blendshape predictions (B, 52)
            - mel_attention_weights: Attention weights from mel stream (optional)
            - emotion_attention_weights: Attention weights from emotion stream (optional)
        """
        batch_size = mel_features.shape[0]
        device = mel_features.device
        
        # Enhanced mel-spectrogram processing with temporal concatenation
        # Long-term context: (B, T, 80) -> (B, 80, T) -> (B, 80, 256)
        mel_features = mel_features.transpose(1, 2)  # (B, 80, T)
        
        # Ensure we have exactly 256 frames (pad or truncate)
        T = mel_features.shape[2]
        if T < self.mel_sequence_length:
            # Pad with zeros
            padding = torch.zeros(
                batch_size, self.num_mel_channels, self.mel_sequence_length - T,
                device=device
            )
            mel_features = torch.cat([mel_features, padding], dim=2)
        elif T > self.mel_sequence_length:
            # Truncate
            mel_features = mel_features[:, :, :self.mel_sequence_length]
        
        # Short-term temporal detail: (B, 3, 80) -> (B, 80, 3)
        mel_temporal_features = mel_temporal_features.transpose(1, 2)  # (B, 80, 3)
        
        # Concatenate long-term and short-term features: (B, 80, 256+3) = (B, 80, 259)
        enhanced_mel_features = torch.cat([mel_features, mel_temporal_features], dim=2)
        
        # Encode enhanced mel channels (each frequency band with temporal detail)
        mel_encoded = self.mel_channel_encoder(enhanced_mel_features)  # (B, 80, d_model)
        mel_encoded = self.mel_norm(mel_encoded)
        
        # Process emotion features
        # Handle both concatenated (B, emotion_dim) and sequential (B, T, emotion_dim) approaches
        if emotion_features.ndim == 2:
            # Concatenated approach: (B, emotion_dim)
            emotion_encoded = self.emotion_encoder(emotion_features)  # (B, d_model)
            emotion_encoded = emotion_encoded.unsqueeze(1)  # (B, 1, d_model) for attention
        else:
            # Sequential approach: (B, T, emotion_dim)
            # For sequential emotion features, we use average pooling to get single representation
            # This maintains compatibility while preserving temporal information
            emotion_pooled = emotion_features.mean(dim=1)  # (B, emotion_dim)
            emotion_encoded = self.emotion_encoder(emotion_pooled)  # (B, d_model)
            emotion_encoded = emotion_encoded.unsqueeze(1)  # (B, 1, d_model) for attention
        
        emotion_encoded = self.emotion_norm(emotion_encoded)
        
        # Prepare queries for each stream
        mouth_queries = self.mouth_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, |mouth|, d_model)
        expression_queries = self.expression_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, |expr|, d_model)
        
        # Mel-stream attention (mouth movements)
        mel_attn_output, mel_attn_weights = self.mel_attention(
            query=mouth_queries,
            key=mel_encoded,
            value=mel_encoded,
            need_weights=return_attention
        )
        mel_attn_output = self.mel_output_proj(mel_attn_output)  # (B, |mouth|, d_model)
        
        # Emotion-stream attention (facial expressions)
        emotion_attn_output, emotion_attn_weights = self.emotion_attention(
            query=expression_queries,
            key=emotion_encoded,
            value=emotion_encoded,
            need_weights=return_attention
        )
        emotion_attn_output = self.emotion_output_proj(emotion_attn_output)  # (B, |expr|, d_model)
        
        # Combine outputs with proper indexing
        combined_features = torch.zeros(batch_size, self.num_blendshapes, self.d_model, device=device)
        combined_features[:, MOUTH_INDICES] = mel_attn_output
        combined_features[:, EXPRESSION_INDICES] = emotion_attn_output
        
        # Decode to blendshapes
        blendshapes = self.blendshape_decoder(combined_features).squeeze(-1)  # (B, 52)
        
        # Natural stream specialization - let learning discover patterns
        # Apply learnable weights without forced specialization
        normalized_mel_weights = F.softmax(self.mel_weights / self.temperature, dim=0)
        normalized_emotion_weights = F.softmax(self.emotion_weights / self.temperature, dim=0)
        
        # Create separate predictions for analysis and visualization
        mel_blendshapes = torch.zeros_like(blendshapes)
        emotion_blendshapes = torch.zeros_like(blendshapes)
        
        # Assign outputs to respective streams for analysis
        mel_blendshapes[:, MOUTH_INDICES] = blendshapes[:, MOUTH_INDICES]
        emotion_blendshapes[:, EXPRESSION_INDICES] = blendshapes[:, EXPRESSION_INDICES]
        
        # Natural weighted combination - let the model learn optimal allocation
        final_blendshapes = (
            normalized_mel_weights * blendshapes * 0.5 +  # Mel stream contribution
            normalized_emotion_weights * blendshapes * 0.5  # Emotion stream contribution
        )
        
        # Clamp to valid range
        final_blendshapes = torch.clamp(final_blendshapes, 0, 1)
        
        output = {'blendshapes': final_blendshapes}
        
        if return_attention:
            output['mel_attention_weights'] = mel_attn_weights  # (B, |mouth|, 80)
            output['emotion_attention_weights'] = emotion_attn_weights  # (B, |expr|, 1)
            output['mel_blendshapes'] = mel_blendshapes
            output['emotion_blendshapes'] = emotion_blendshapes
            
        return output
    
    def get_frequency_bands(self) -> Dict[str, List[int]]:
        """
        Get frequency band groupings for visualization.
        
        Returns:
            Dictionary mapping frequency band names to mel channel indices
        """
        return {
            'low': list(range(0, 20)),      # 0-20: Low frequencies (voice fundamental)
            'mid_low': list(range(20, 40)), # 20-40: Lower-mid frequencies
            'mid_high': list(range(40, 60)),# 40-60: Upper-mid frequencies  
            'high': list(range(60, 80))     # 60-80: High frequencies (consonants)
        }


class DualStreamEncoder(nn.Module):
    """
    Encoder for dual-stream architecture that processes mel and emotion features independently.
    """
    
    def __init__(
        self,
        mel_dim: int = 80,
        emotion_dim: int = 256,  # Concatenated eGeMAPS dimension
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize dual-stream encoder.
        
        Args:
            mel_dim: Mel-spectrogram dimension
            emotion_dim: Concatenated eGeMAPS dimension (256)
            d_model: Model dimension
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Mel-stream encoder (no fusion needed)
        self.mel_encoder = nn.Sequential(
            nn.Linear(mel_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Emotion-stream encoder  
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Optional transformer layers for each stream
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.mel_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.emotion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.mel_transformer = None
            self.emotion_transformer = None
            
    def forward(
        self,
        mel_features: torch.Tensor,
        emotion_features: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
        emotion_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of dual-stream encoder.
        
        Args:
            mel_features: Mel features (B, T, mel_dim)
            emotion_features: Emotion features (B, T, emotion_dim)
            mel_mask: Mask for mel features (B, T)
            emotion_mask: Mask for emotion features (B, T)
            
        Returns:
            Tuple of (mel_encoded, emotion_encoded), both of shape (B, T, d_model)
        """
        # Encode mel features
        mel_encoded = self.mel_encoder(mel_features)
        if self.mel_transformer is not None:
            mel_encoded = self.mel_transformer(
                mel_encoded, 
                src_key_padding_mask=~mel_mask if mel_mask is not None else None
            )
        
        # Encode emotion features
        emotion_encoded = self.emotion_encoder(emotion_features)
        if self.emotion_transformer is not None:
            emotion_encoded = self.emotion_transformer(
                emotion_encoded,
                src_key_padding_mask=~emotion_mask if emotion_mask is not None else None
            )
            
        return mel_encoded, emotion_encoded