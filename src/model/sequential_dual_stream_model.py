"""
Sequential Dual-Stream Model for time-series blendshape generation.

This model extends SimplifiedDualStreamModel to output full sequences
instead of single frames, supporting both 30fps and 60fps.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging
from .simplified_dual_stream_model import SimplifiedDualStreamModel

logger = logging.getLogger(__name__)


class SequentialDualStreamModel(SimplifiedDualStreamModel):
    """
    Sequential version of dual-stream model that outputs full sequences.
    
    Processes audio in sliding windows to generate frame-by-frame blendshapes,
    maintaining temporal consistency across the sequence.
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
        device: str = "cuda",
        real_time_mode: bool = False,
        stride_frames: int = 1,  # Number of frames to stride between windows
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_blendshapes=num_blendshapes,
            sample_rate=sample_rate,
            target_fps=target_fps,
            mel_sequence_length=mel_sequence_length,
            emotion_config=emotion_config,
            device=device,
            real_time_mode=real_time_mode,
        )
        
        self.stride_frames = stride_frames
        self.window_frames = mel_sequence_length  # Use mel_sequence_length as window size
        
        # Calculate window parameters in samples
        self.window_samples = self.window_frames * self.hop_length
        self.stride_samples = self.stride_frames * self.hop_length
        
        logger.info(f"Sequential model initialized:")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Window frames: {self.window_frames}")
        logger.info(f"  Stride frames: {self.stride_frames}")
        logger.info(f"  Hop length: {self.hop_length}")
    
    def forward(
        self, 
        audio: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass generating full sequence of blendshapes.
        
        Args:
            audio: Audio tensor of shape (B, T)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - blendshapes: Sequence of blendshapes (B, T_out, 52)
            - mel_attention_weights: Attention weights (optional)
            - emotion_attention_weights: Attention weights (optional)
        """
        batch_size, audio_length = audio.shape
        
        # Calculate output sequence length
        num_frames = audio_length // self.hop_length
        
        # Extract emotion features ONCE for the entire audio
        # This is much more efficient than extracting per window
        emotion_features, emotion_metadata = self.extract_emotion_features(audio)
        
        # Process mel features in sliding windows
        all_blendshapes = []
        all_mel_attentions = [] if return_attention else None
        all_emotion_attentions = [] if return_attention else None
        
        # Calculate number of output frames based on stride
        num_output_frames = max(1, (num_frames - self.window_frames) // self.stride_frames + 1)
        
        # Reset temporal state at beginning of sequence
        self.reset_temporal_state()
        
        for i in range(num_output_frames):
            # Calculate frame indices
            start_frame = i * self.stride_frames
            end_frame = start_frame + self.window_frames
            
            # Extract audio window
            start_sample = start_frame * self.hop_length
            end_sample = min(end_frame * self.hop_length, audio_length)
            
            # Handle last window that might be shorter
            if end_sample - start_sample < self.window_samples:
                # Pad the last window
                window_audio = torch.zeros(batch_size, self.window_samples, device=audio.device)
                actual_length = end_sample - start_sample
                window_audio[:, :actual_length] = audio[:, start_sample:end_sample]
            else:
                window_audio = audio[:, start_sample:end_sample]
            
            # Extract mel features for this window
            mel_features, mel_temporal_features = self.extract_mel_features(window_audio)
            
            # Align features (emotion features are already extracted)
            mel_features_aligned, emotion_features_aligned = self.align_features(
                mel_features, emotion_features
            )
            
            # Apply dual-stream attention
            output = self.dual_stream_attention(
                mel_features=mel_features_aligned,
                mel_temporal_features=mel_temporal_features,
                emotion_features=emotion_features_aligned,
                return_attention=return_attention,
            )
            
            # Apply temporal smoothing
            smoothed_blendshapes = self.apply_temporal_smoothing(output['blendshapes'])
            
            # Collect outputs
            all_blendshapes.append(smoothed_blendshapes)
            
            if return_attention:
                if 'mel_attention_weights' in output:
                    all_mel_attentions.append(output['mel_attention_weights'])
                if 'emotion_attention_weights' in output:
                    all_emotion_attentions.append(output['emotion_attention_weights'])
        
        # Stack all outputs
        results = {}
        
        # Stack blendshapes: (B, num_output_frames, 52)
        blendshapes_sequence = torch.stack(all_blendshapes, dim=1)
        results['blendshapes'] = blendshapes_sequence
        
        # Include attention weights if requested
        if return_attention:
            if all_mel_attentions:
                results['mel_attention_weights'] = torch.stack(all_mel_attentions, dim=1)
            if all_emotion_attentions:
                results['emotion_attention_weights'] = torch.stack(all_emotion_attentions, dim=1)
        
        # Add metadata
        results['num_frames'] = blendshapes_sequence.shape[1]
        results['fps'] = self.target_fps
        results['emotion_backend'] = emotion_metadata.get("backend_used", "unknown")
        results['emotion_processing_time'] = emotion_metadata.get("processing_time", 0.0)
        
        return results
    
    def forward_single_frame(self, audio: torch.Tensor, frame_idx: int = None) -> Dict[str, torch.Tensor]:
        """
        Process a single frame (for compatibility with real-time inference).
        
        Args:
            audio: Audio tensor of shape (B, T)
            frame_idx: Optional frame index to process (default: middle frame)
            
        Returns:
            Single frame output compatible with parent class
        """
        # Use parent class for single frame processing
        return super().forward(audio, return_attention=False)