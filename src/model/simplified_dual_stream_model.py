"""
Simplified dual-stream KoeMorph model for training.

This model implements the dual-stream architecture where mel-spectrograms 
control mouth movements and emotion2vec controls facial expressions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging

from .dual_stream_attention import DualStreamCrossAttention
from ..features.emotion_extractor import EmotionExtractor
from ..features.mel_sliding_window import MelSlidingWindowExtractor

logger = logging.getLogger(__name__)


class SimplifiedDualStreamModel(nn.Module):
    """
    Simplified dual-stream model that processes mel-spectrograms and emotion features separately.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_blendshapes: int = 52,
        sample_rate: int = 16000,
        target_fps: int = 30,
        mel_sequence_length: int = 256,
        emotion_config: Optional[Dict] = None,
        mel_config: Optional[Dict] = None,
        device: str = "cuda",
        real_time_mode: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_blendshapes = num_blendshapes
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.mel_sequence_length = mel_sequence_length
        self.device = device
        self.real_time_mode = real_time_mode
        
        # Audio processing parameters
        self.n_mels = 80
        # Use consistent hop_length calculation: sample_rate / target_fps
        self.hop_length = int(sample_rate / target_fps)  # 533.33... -> 533 for 16kHz, 30fps
        self.n_fft = 1024
        
        # Initialize emotion extractor with robust fallback
        if emotion_config is None:
            emotion_config = {
                "backend": "emotion2vec",
                "model_name": "iic/emotion2vec_plus_large",
                "device": device,
                "sample_rate": sample_rate,
                "enable_caching": True,
                "batch_size": 4,
            }
        
        logger.info("Initializing emotion extractor...")
        
        # Split emotion_config into EmotionExtractor params and OpenSMILE params
        emotion_extractor_params = {
            key: value for key, value in emotion_config.items()
            if key in ['backend', 'model_name', 'device', 'cache_dir', 'enable_caching', 'batch_size', 'sample_rate']
        }
        
        # Store OpenSMILE-specific params for later use
        self.opensmile_config = {
            key: value for key, value in emotion_config.items()
            if key not in emotion_extractor_params
        }
        
        self.emotion_extractor = EmotionExtractor(**emotion_extractor_params)
        
        # Pass OpenSMILE config to the emotion extractor for proper initialization
        if hasattr(self.emotion_extractor, '_initialize_opensmile'):
            self.emotion_extractor._opensmile_config = self.opensmile_config
        
        # Force fallback to OpenSMILE with concatenation if backend is "opensmile"
        if emotion_extractor_params.get("backend") == "opensmile":
            logger.info("Forcing OpenSMILE backend initialization...")
            self.emotion_extractor.fallback_level = 1
            self.emotion_extractor._opensmile_config = self.opensmile_config
            self.emotion_extractor._initialize_opensmile()
        
        # Determine emotion dimension based on the backend used
        if self.emotion_extractor.fallback_level == 0:  # emotion2vec
            self.emotion_dim = 1024
        elif self.emotion_extractor.fallback_level == 1:  # opensmile eGeMAPS
            # Check if using concatenation approach (PRODUCTION)
            if hasattr(self.emotion_extractor.opensmile_extractor, 'use_concatenation') and \
               self.emotion_extractor.opensmile_extractor.use_concatenation:
                self.emotion_dim = 256  # Concatenated + compressed dimension
                logger.info("Using concatenated eGeMAPS approach: 3×88→256")
            else:
                # Legacy: get actual feature dimension from OpenSMILE extractor
                if hasattr(self.emotion_extractor.opensmile_extractor, 'feature_dim'):
                    self.emotion_dim = self.emotion_extractor.opensmile_extractor.feature_dim
                else:
                    self.emotion_dim = 88  # eGeMAPS standard feature count
        else:  # basic
            self.emotion_dim = 9  # basic prosodic features
        
        logger.info(f"Using emotion backend: {self.emotion_extractor._get_backend_name()}")
        logger.info(f"Emotion feature dimension: {self.emotion_dim}")
        
        # Initialize mel-spectrogram extractor
        logger.info("Initializing mel-spectrogram extractor...")
        mel_config = mel_config or {}
        
        if self.real_time_mode:
            # Use sliding window extractor for real-time processing
            self.mel_extractor = MelSlidingWindowExtractor(
                context_window=mel_config.get("context_window", 8.5),  # 8.5 seconds
                update_interval=mel_config.get("update_interval", 0.0333),  # 33.3ms
                sample_rate=sample_rate,
                n_mels=self.n_mels,
                n_fft=mel_config.get("n_fft", 1024),
                hop_length=self.hop_length,
                f_min=mel_config.get("f_min", 80.0),
                f_max=mel_config.get("f_max", sample_rate // 2),
                device=device,
                **{k: v for k, v in mel_config.items() if k not in [
                    'context_window', 'update_interval', 'n_fft', 'f_min', 'f_max'
                ]}
            )
            self.mel_context_window = mel_config.get("context_window", 8.5)
            self.mel_update_interval = mel_config.get("update_interval", 0.0333)
            logger.info(f"Real-time mel extractor initialized:")
            logger.info(f"  Context window: {self.mel_context_window}s")
            logger.info(f"  Update interval: {self.mel_update_interval*1000:.1f}ms")
        else:
            # Use standard batch processing for training
            self.mel_extractor = None
            
        logger.info(f"Mel processing mode: {'Real-time' if self.real_time_mode else 'Batch'}")
            
        # Enhanced dual-stream cross-attention module
        self.dual_stream_attention = DualStreamCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_mel_channels=self.n_mels,
            mel_sequence_length=mel_sequence_length,
            mel_temporal_frames=3,  # 3 frames for temporal detail
            emotion_dim=self.emotion_dim,
            dropout=0.1,
            num_blendshapes=num_blendshapes,
            use_learnable_weights=True,
            temperature=1.0,
        )
        
        # Temporal smoothing (optional)
        self.use_temporal_smoothing = True
        self.smoothing_alpha = nn.Parameter(torch.tensor(0.8))
        self.prev_blendshapes = None
        
    def extract_mel_features(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract enhanced mel-spectrogram features from audio.
        
        Args:
            audio: Audio tensor of shape (B, T)
            
        Returns:
            Tuple of:
            - Long-term mel features of shape (B, T_mel, 80)
            - Short-term mel features of shape (B, 3, 80) for temporal detail
        """
        batch_size, audio_length = audio.shape
        
        # Convert to numpy for librosa processing
        long_term_features = []
        short_term_features = []
        
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            
            # Extract full mel-spectrogram for long-term context
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
            mel = (mel + 80) / 80  # Normalize to approximately [0, 1]
            
            mel_T = mel.T  # (T_mel, 80)
            long_term_features.append(mel_T)
            
            # Extract short-term temporal detail (last 3 frames)
            if mel_T.shape[0] >= 3:
                temporal_detail = mel_T[-3:]  # (3, 80) - last 3 frames
            else:
                # Pad if we don't have enough frames
                temporal_detail = np.zeros((3, self.n_mels))
                if mel_T.shape[0] > 0:
                    temporal_detail[:mel_T.shape[0]] = mel_T
            
            short_term_features.append(temporal_detail)
        
        # Stack long-term features
        max_length = max(feat.shape[0] for feat in long_term_features)
        long_term_padded = np.zeros((batch_size, max_length, self.n_mels))
        
        for i, feat in enumerate(long_term_features):
            long_term_padded[i, :feat.shape[0], :] = feat
        
        # Stack short-term features
        short_term_stacked = np.stack(short_term_features)  # (B, 3, 80)
        
        return (
            torch.tensor(long_term_padded, dtype=torch.float32, device=audio.device),
            torch.tensor(short_term_stacked, dtype=torch.float32, device=audio.device)
        )
    
    def extract_emotion_features(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Extract emotion features using the robust emotion extractor.
        
        Args:
            audio: Audio tensor of shape (B, T)
            
        Returns:
            Tuple of:
            - Emotion features of shape (B, emotion_dim) for concatenated or (B, T_emotion, emotion_dim) for sequential
            - Metadata dictionary with extraction info
        """
        # Extract emotion features using the robust extractor
        results = self.emotion_extractor.extract_features(
            audio,
            return_embeddings=True,
            return_predictions=True
        )
        
        if results["embeddings"] is None or len(results["embeddings"]) == 0:
            # Create dummy features if extraction failed
            batch_size = audio.shape[0]
            logger.warning("Emotion extraction failed, using dummy features")
            
            if hasattr(self.emotion_extractor.opensmile_extractor, 'use_concatenation') and \
               self.emotion_extractor.opensmile_extractor.use_concatenation:
                # Concatenated approach: return (B, 256)
                dummy_features = torch.randn(
                    batch_size, self.emotion_dim, device=audio.device
                ) * 0.1
                return dummy_features, {"backend_used": "dummy", "extraction_failed": True}
            else:
                # Sequential approach: return (B, 1, emotion_dim)
                dummy_features = torch.randn(
                    batch_size, 1, self.emotion_dim, device=audio.device
                ) * 0.1
                return dummy_features, {"backend_used": "dummy", "extraction_failed": True}
        
        # Convert embeddings to tensor
        emotion_embeddings = torch.tensor(
            results["embeddings"], 
            dtype=torch.float32, 
            device=audio.device
        )
        
        # Check if using concatenated approach
        if hasattr(self.emotion_extractor.opensmile_extractor, 'use_concatenation') and \
           self.emotion_extractor.opensmile_extractor.use_concatenation:
            # Concatenated approach: return features as (B, 256)
            if emotion_embeddings.ndim == 2:  # Already (B, emotion_dim)
                return emotion_embeddings, results["metadata"]
            elif emotion_embeddings.ndim == 1:  # Single sample (emotion_dim,)
                return emotion_embeddings.unsqueeze(0), results["metadata"]  # (1, emotion_dim)
            else:
                # Unexpected shape, flatten to (B, emotion_dim)
                batch_size = audio.shape[0]
                emotion_embeddings = emotion_embeddings.view(batch_size, -1)
                return emotion_embeddings, results["metadata"]
        else:
            # Sequential approach: ensure proper shape (B, T_emotion, emotion_dim)
            if emotion_embeddings.ndim == 2:  # (B, emotion_dim)
                emotion_embeddings = emotion_embeddings.unsqueeze(1)  # (B, 1, emotion_dim)
            
            # For utterance-level features, we might want to replicate across time
            # to match mel-spectrogram temporal resolution
            target_seq_len = max(1, int(audio.shape[1] / self.sample_rate * 2))  # ~2 Hz
            if emotion_embeddings.shape[1] == 1 and target_seq_len > 1:
                emotion_embeddings = emotion_embeddings.repeat(1, target_seq_len, 1)
            
            return emotion_embeddings, results["metadata"]
    
    def align_features(
        self, 
        mel_features: torch.Tensor, 
        emotion_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align mel and emotion features to have compatible temporal dimensions.
        
        Args:
            mel_features: Mel features (B, T_mel, 80)
            emotion_features: Emotion features (B, emotion_dim) for concatenated or (B, T_emotion, emotion_dim) for sequential
            
        Returns:
            Aligned features with same temporal dimension
        """
        # Check if using concatenated approach
        if hasattr(self.emotion_extractor.opensmile_extractor, 'use_concatenation') and \
           self.emotion_extractor.opensmile_extractor.use_concatenation:
            # Concatenated approach: emotion_features is (B, 256), no alignment needed
            # The dual_stream_attention expects (B, emotion_dim) for concatenated features
            return mel_features, emotion_features
        else:
            # Sequential approach: align temporal dimensions
            B, T_mel, _ = mel_features.shape
            B_e, T_emotion, _ = emotion_features.shape
            
            if T_emotion != T_mel:
                # Interpolate emotion features to match mel timeline
                emotion_features = emotion_features.transpose(1, 2)  # (B, emotion_dim, T_emotion)
                emotion_features = F.interpolate(
                    emotion_features,
                    size=T_mel,
                    mode='linear',
                    align_corners=False
                )
                emotion_features = emotion_features.transpose(1, 2)  # (B, T_mel, emotion_dim)
                
            return mel_features, emotion_features
    
    def apply_temporal_smoothing(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """
        Apply exponential smoothing to blendshapes for temporal consistency.
        
        Args:
            blendshapes: Current blendshapes (B, 52)
            
        Returns:
            Smoothed blendshapes (B, 52)
        """
        if not self.use_temporal_smoothing:
            return blendshapes
            
        batch_size = blendshapes.shape[0]
        
        # Check if previous blendshapes exist and have the correct batch size
        if self.prev_blendshapes is None or self.prev_blendshapes.shape[0] != batch_size:
            self.prev_blendshapes = blendshapes.detach()
            return blendshapes
        
        # Exponential moving average
        alpha = torch.sigmoid(self.smoothing_alpha)  # Ensure alpha is in [0, 1]
        smoothed = alpha * blendshapes + (1 - alpha) * self.prev_blendshapes
        
        # Update previous blendshapes
        self.prev_blendshapes = smoothed.detach()
        
        return smoothed
    
    def forward(
        self, 
        audio: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of dual-stream model.
        
        Args:
            audio: Audio tensor of shape (B, T)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - blendshapes: Final blendshape predictions (B, 52)
            - mel_attention_weights: Attention from mel stream (optional)
            - emotion_attention_weights: Attention from emotion stream (optional)
        """
        # Initialize results dictionary
        results = {}
        
        # Extract enhanced features
        mel_features, mel_temporal_features = self.extract_mel_features(audio)  # (B, T_mel, 80), (B, 3, 80)
        
        # Extract emotion features with metadata
        emotion_features, emotion_metadata = self.extract_emotion_features(audio)
        
        # Store metadata for debugging
        results["emotion_backend"] = emotion_metadata.get("backend_used", "unknown")
        results["emotion_processing_time"] = emotion_metadata.get("processing_time", 0.0)
        
        # Align temporal dimensions
        mel_features, emotion_features = self.align_features(mel_features, emotion_features)
        
        # Apply enhanced dual-stream attention
        output = self.dual_stream_attention(
            mel_features=mel_features,
            mel_temporal_features=mel_temporal_features,
            emotion_features=emotion_features,
            return_attention=return_attention,
        )
        
        # Apply temporal smoothing
        output['blendshapes'] = self.apply_temporal_smoothing(output['blendshapes'])
        
        return output
    
    def reset_temporal_state(self):
        """Reset temporal state for new sequence."""
        self.prev_blendshapes = None
        
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        # Get emotion extractor statistics
        emotion_stats = self.emotion_extractor.get_statistics()
        
        info = {
            'model_type': 'SimplifiedDualStreamModel',
            'd_model': self.d_model,
            'num_heads': self.dual_stream_attention.num_heads,
            'num_blendshapes': self.num_blendshapes,
            'emotion_backend': self.emotion_extractor._get_backend_name(),
            'emotion_fallback_level': self.emotion_extractor.fallback_level,
            'mel_sequence_length': self.mel_sequence_length,
            'n_mels': self.n_mels,
            'emotion_dim': self.emotion_dim,
            'total_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'emotion_extraction_stats': emotion_stats,
            'real_time_mode': self.real_time_mode,
        }
        
        # Add mel extractor info if in real-time mode
        if self.real_time_mode and self.mel_extractor:
            mel_stats = self.mel_extractor.get_stats()
            info.update({
                'mel_context_window': self.mel_context_window,
                'mel_update_interval': self.mel_update_interval,
                'mel_extraction_stats': mel_stats,
            })
        
        return info
    
    def process_audio_frame_realtime(
        self, 
        audio_frame: np.ndarray,
        return_attention: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Process single audio frame in real-time mode.
        
        Args:
            audio_frame: Audio frame samples (should match hop_length)
            return_attention: Whether to return attention weights
            
        Returns:
            Blendshape predictions (52,) or None if not ready
        """
        if not self.real_time_mode:
            raise RuntimeError("Model not in real-time mode. Use forward() for batch processing.")
        
        if self.mel_extractor is None:
            raise RuntimeError("Mel extractor not initialized for real-time mode.")
        
        # Process mel features
        mel_features = self.mel_extractor.process_audio_frame(audio_frame)
        if mel_features is None:
            return None  # Not enough context yet
        
        # Convert to torch tensor
        mel_features = torch.from_numpy(mel_features).unsqueeze(0).to(self.device)  # (1, T, 80)
        
        # Process emotion features (using the same audio frame for now)
        # In a real system, emotion features would be updated at their own interval
        emotion_audio = torch.from_numpy(audio_frame).unsqueeze(0).to(self.device)  # (1, hop_length)
        emotion_features, _ = self.extract_emotion_features(emotion_audio)
        
        # Align features
        mel_features, emotion_features = self.align_features(mel_features, emotion_features)
        
        # Apply dual-stream attention
        with torch.no_grad():
            output = self.dual_stream_attention(
                mel_features=mel_features,
                emotion_features=emotion_features,
                return_attention=return_attention,
            )
        
        # Apply temporal smoothing
        blendshapes = self.apply_temporal_smoothing(output['blendshapes'])
        
        return blendshapes.squeeze(0)  # Return (52,)
    
    def reset_realtime_state(self):
        """Reset all real-time processing state."""
        if self.real_time_mode and self.mel_extractor:
            self.mel_extractor.reset()
        self.reset_temporal_state()
        logger.info("Real-time processing state reset")
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics."""
        if not self.real_time_mode:
            return {"error": "Not in real-time mode"}
        
        stats = {}
        
        # Mel extractor stats
        if self.mel_extractor:
            stats["mel_stats"] = self.mel_extractor.get_stats()
        
        # Emotion extractor stats
        stats["emotion_stats"] = self.emotion_extractor.get_statistics()
        
        return stats