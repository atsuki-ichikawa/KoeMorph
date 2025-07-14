"""
Main GaussianFace model combining all components.

Unit name     : GaussianFaceModel
Input         : Audio features (mel, prosody, emotion2vec)
Output        : ARKit 52 blendshapes [0,1]
Dependencies  : All model components
Assumptions   : Synchronized multi-stream input
Failure modes : Feature dimension mismatch, memory issues
Test cases    : test_end_to_end, test_real_time_inference, test_model_saving
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .attention import (
    BlendshapeQueryEmbedding,
    MultiHeadCrossAttention,
    MultiStreamAudioEncoder,
)
from .decoder import BlendshapeDecoder, BlendshapeConstraints, TemporalSmoother
from .losses import GaussianFaceLoss


class GaussianFaceModel(nn.Module):
    """
    Complete GaussianFace model for real-time blendshape generation.
    
    Combines multi-stream audio encoding, cross-attention, and blendshape
    decoding with temporal smoothing and constraints.
    """
    
    def __init__(
        self,
        # Audio encoder settings
        mel_dim: int = 80,
        prosody_dim: int = 4,
        emotion_dim: int = 256,
        
        # Model architecture
        d_model: int = 256,
        d_query: int = 128,
        d_key: int = 256,
        d_value: int = 256,
        
        # Attention settings
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        num_attention_layers: int = 4,
        attention_dropout: float = 0.1,
        
        # Decoder settings
        decoder_hidden_dim: int = 128,
        decoder_layers: int = 2,
        decoder_activation: str = "gelu",
        output_activation: str = "sigmoid",
        
        # Temporal processing
        use_temporal_smoothing: bool = True,
        smoothing_method: str = "exponential",
        smoothing_alpha: float = 0.8,
        
        # Constraints
        use_constraints: bool = True,
        
        # Real-time settings
        causal: bool = True,
        window_size: Optional[int] = 30,
        
        # Other settings
        num_blendshapes: int = 52,
        dropout: float = 0.1,
    ):
        """
        Initialize GaussianFace model.
        
        Args:
            mel_dim: Mel-spectrogram feature dimension
            prosody_dim: Prosody feature dimension  
            emotion_dim: Emotion2vec feature dimension
            d_model: Model hidden dimension
            d_query: Query dimension for attention
            d_key: Key dimension for attention
            d_value: Value dimension for attention
            num_heads: Number of attention heads
            num_encoder_layers: Number of audio encoder layers
            num_attention_layers: Number of cross-attention layers
            attention_dropout: Attention dropout rate
            decoder_hidden_dim: Decoder hidden dimension
            decoder_layers: Number of decoder layers
            decoder_activation: Decoder activation function
            output_activation: Final output activation
            use_temporal_smoothing: Whether to apply temporal smoothing
            smoothing_method: Temporal smoothing method
            smoothing_alpha: Smoothing factor
            use_constraints: Whether to apply blendshape constraints
            causal: Whether to use causal attention
            window_size: Attention window size for real-time
            num_blendshapes: Number of output blendshapes
            dropout: General dropout rate
        """
        super().__init__()
        
        self.mel_dim = mel_dim
        self.prosody_dim = prosody_dim
        self.emotion_dim = emotion_dim
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        self.use_temporal_smoothing = use_temporal_smoothing
        self.use_constraints = use_constraints
        
        # Multi-stream audio encoder
        self.audio_encoder = MultiStreamAudioEncoder(
            mel_dim=mel_dim,
            prosody_dim=prosody_dim,
            emotion_dim=emotion_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            dropout=dropout,
            fusion_method="concat",
            use_positional_encoding=True,
        )
        
        # Blendshape query embeddings
        self.query_embeddings = BlendshapeQueryEmbedding(
            num_blendshapes=num_blendshapes,
            d_query=d_query,
            use_conditioning=True,
            dropout=dropout,
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadCrossAttention(
                d_query=d_query,
                d_key=d_model,
                d_value=d_model,
                d_model=d_model,
                num_heads=num_heads,
                dropout=attention_dropout,
                causal=causal,
                window_size=window_size,
            )
            for _ in range(num_attention_layers)
        ])
        
        # Layer normalization for attention layers
        self.attention_layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_attention_layers)
        ])
        
        # Blendshape decoder
        self.decoder = BlendshapeDecoder(
            d_model=d_model,
            hidden_dim=decoder_hidden_dim,
            num_blendshapes=num_blendshapes,
            num_layers=decoder_layers,
            activation=decoder_activation,
            output_activation=output_activation,
            dropout=dropout,
            use_residual=True,
            use_layer_norm=True,
        )
        
        # Temporal smoothing
        if use_temporal_smoothing:
            self.temporal_smoother = TemporalSmoother(
                num_blendshapes=num_blendshapes,
                smoothing_method=smoothing_method,
                alpha=smoothing_alpha,
                learnable=True,
            )
        
        # Blendshape constraints
        if use_constraints:
            self.constraints = BlendshapeConstraints(
                num_blendshapes=num_blendshapes,
                mutual_exclusions=None,  # Use defaults
                smoothness_weight=0.1,
            )
    
    def forward(
        self,
        mel_features: torch.Tensor,
        prosody_features: torch.Tensor,
        emotion_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        prev_blendshapes: Optional[torch.Tensor] = None,
        apply_smoothing: bool = True,
        apply_constraints: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GaussianFace model.
        
        Args:
            mel_features: Mel-spectrogram features (B, T, mel_dim)
            prosody_features: Prosody features (B, T, prosody_dim)
            emotion_features: Emotion2vec features (B, T, emotion_dim)
            audio_mask: Audio padding mask (B, T)
            prev_blendshapes: Previous blendshape state (B, 52)
            apply_smoothing: Whether to apply temporal smoothing
            apply_constraints: Whether to apply constraints
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - blendshapes: Final blendshape coefficients (B, 52)
            - raw_blendshapes: Pre-smoothing/constraint blendshapes (B, 52)
            - attention_weights: Attention weights if requested
        """
        batch_size = mel_features.shape[0]
        device = mel_features.device
        
        # Encode multi-stream audio features
        encoded_audio = self.audio_encoder(
            mel_features, prosody_features, emotion_features, mask=audio_mask
        )  # (B, T, d_model)
        
        # Generate query embeddings for blendshapes
        query_embeddings = self.query_embeddings(
            batch_size, prev_blendshapes
        )  # (B, 52, d_query)
        
        # Apply cross-attention layers
        attention_output = query_embeddings
        attention_weights_list = []
        
        for i, (attention_layer, layer_norm) in enumerate(
            zip(self.cross_attention_layers, self.attention_layer_norms)
        ):
            # Cross-attention
            attn_out, attn_weights = attention_layer(
                query=attention_output,
                key=encoded_audio,
                value=encoded_audio,
                key_padding_mask=audio_mask,
                return_attention=return_attention,
            )
            
            # Residual connection + layer norm
            attention_output = layer_norm(attn_out + attention_output)
            
            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)
        
        # Decode to blendshapes
        raw_blendshapes = self.decoder(attention_output, prev_blendshapes)
        
        # Apply temporal smoothing
        if apply_smoothing and self.use_temporal_smoothing:
            smoothed_blendshapes = self.temporal_smoother(raw_blendshapes)
        else:
            smoothed_blendshapes = raw_blendshapes
        
        # Apply constraints
        if apply_constraints and self.use_constraints:
            final_blendshapes, constraint_violations = self.constraints(
                smoothed_blendshapes, apply_constraints=True, return_violations=False
            )
        else:
            final_blendshapes = smoothed_blendshapes
        
        # Prepare output
        output = {
            'blendshapes': final_blendshapes,
            'raw_blendshapes': raw_blendshapes,
        }
        
        if return_attention and attention_weights_list:
            output['attention_weights'] = attention_weights_list
        
        return output
    
    def reset_temporal_state(self):
        """Reset temporal state for new sequence."""
        if hasattr(self, 'temporal_smoother'):
            self.temporal_smoother._reset_state(next(self.parameters()).device)
        
        if hasattr(self, 'constraints'):
            self.constraints.reset_state()
    
    def inference_step(
        self,
        mel_features: torch.Tensor,
        prosody_features: torch.Tensor,
        emotion_features: torch.Tensor,
        prev_blendshapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single inference step for real-time processing.
        
        Args:
            mel_features: Mel features for current frame (1, 1, mel_dim)
            prosody_features: Prosody features for current frame (1, 1, prosody_dim) 
            emotion_features: Emotion features for current frame (1, 1, emotion_dim)
            prev_blendshapes: Previous blendshape state (1, 52)
            
        Returns:
            Current blendshape coefficients (1, 52)
        """
        with torch.no_grad():
            output = self.forward(
                mel_features,
                prosody_features, 
                emotion_features,
                prev_blendshapes=prev_blendshapes,
                apply_smoothing=True,
                apply_constraints=True,
                return_attention=False,
            )
            
            return output['blendshapes']
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        return {
            'total_parameters': self.get_num_parameters(),
            'mel_dim': self.mel_dim,
            'prosody_dim': self.prosody_dim,
            'emotion_dim': self.emotion_dim,
            'd_model': self.d_model,
            'num_blendshapes': self.num_blendshapes,
            'num_attention_layers': len(self.cross_attention_layers),
            'use_temporal_smoothing': self.use_temporal_smoothing,
            'use_constraints': self.use_constraints,
        }


def create_gaussian_face_model(config: dict) -> GaussianFaceModel:
    """
    Create GaussianFace model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized GaussianFace model
    """
    model = GaussianFaceModel(
        # Audio dimensions
        mel_dim=config.get('mel_dim', 80),
        prosody_dim=config.get('prosody_dim', 4),
        emotion_dim=config.get('emotion_dim', 256),
        
        # Architecture
        d_model=config.get('d_model', 256),
        d_query=config.get('d_query', 128),
        d_key=config.get('d_key', 256),
        d_value=config.get('d_value', 256),
        
        # Attention
        num_heads=config.get('num_heads', 8),
        num_encoder_layers=config.get('num_encoder_layers', 2),
        num_attention_layers=config.get('num_attention_layers', 4),
        attention_dropout=config.get('attention_dropout', 0.1),
        
        # Decoder
        decoder_hidden_dim=config.get('decoder_hidden_dim', 128),
        decoder_layers=config.get('decoder_layers', 2),
        decoder_activation=config.get('decoder_activation', 'gelu'),
        output_activation=config.get('output_activation', 'sigmoid'),
        
        # Temporal processing
        use_temporal_smoothing=config.get('use_temporal_smoothing', True),
        smoothing_method=config.get('smoothing_method', 'exponential'),
        smoothing_alpha=config.get('smoothing_alpha', 0.8),
        
        # Constraints
        use_constraints=config.get('use_constraints', True),
        
        # Real-time
        causal=config.get('causal', True),
        window_size=config.get('window_size', 30),
        
        # General
        num_blendshapes=config.get('num_blendshapes', 52),
        dropout=config.get('dropout', 0.1),
    )
    
    return model