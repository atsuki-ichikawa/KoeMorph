"""
Real-time mel-spectrogram extractor with sliding window processing.

This module provides efficient sliding window processing for mel-spectrograms,
matching the design of OpenSMILE eGeMAPS but optimized for mel features.
"""

import logging
import time
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import librosa
from collections import deque
import threading

logger = logging.getLogger(__name__)


class MelAudioBuffer:
    """
    Circular audio buffer for efficient sliding window mel-spectrogram processing.
    
    Maintains a fixed-size audio buffer and provides efficient frame-by-frame updates.
    """
    
    def __init__(
        self,
        context_window: float = 8.5,  # seconds
        sample_rate: int = 16000,
        update_interval: float = 0.0333,  # 33.3ms = 30fps
    ):
        """
        Initialize mel audio buffer.
        
        Args:
            context_window: Audio context window in seconds (8.5s)
            sample_rate: Audio sample rate (16kHz)
            update_interval: Frame update interval in seconds (33.3ms)
        """
        self.context_window = context_window
        self.sample_rate = sample_rate
        self.update_interval = update_interval
        
        # Calculate buffer parameters
        self.buffer_size = int(context_window * sample_rate)  # 136,000 samples
        # Unified hop_length calculation: int(sample_rate / target_fps) = 533
        target_fps = 1.0 / update_interval  # 30 FPS from 0.0333s interval
        self.hop_length = int(sample_rate / target_fps)  # 533 samples for 16kHz/30fps
        
        # Initialize circular buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.write_ptr = 0
        self.is_full = False
        
        # Threading lock for thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.total_frames_added = 0
        self.buffer_overruns = 0
        
        logger.info(f"Mel audio buffer initialized:")
        logger.info(f"  Context window: {context_window}s")
        logger.info(f"  Buffer size: {self.buffer_size} samples")
        logger.info(f"  Hop length: {self.hop_length} samples ({update_interval*1000:.1f}ms)")
        logger.info(f"  Expected frames: ~{int(context_window / update_interval)}")
    
    def add_audio_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Add new audio frame to the circular buffer.
        
        Args:
            audio_frame: Audio frame samples (should be hop_length samples)
            
        Returns:
            True if frame was added successfully
        """
        frame_len = len(audio_frame)
        
        # Allow ±1 sample tolerance for frame size
        if abs(frame_len - self.hop_length) > 1:
            logger.warning(f"Frame size mismatch: expected ~{self.hop_length}, got {frame_len}")
            return False
        
        # Handle frame size adjustment
        if frame_len < self.hop_length:
            # Pad with zeros
            audio_frame = np.pad(audio_frame, (0, self.hop_length - frame_len), mode='constant')
        elif frame_len > self.hop_length:
            # Truncate
            audio_frame = audio_frame[:self.hop_length]
        
        with self._lock:
            # Add frame to circular buffer
            end_ptr = (self.write_ptr + self.hop_length) % self.buffer_size
            
            if end_ptr > self.write_ptr:
                # No wraparound
                self.audio_buffer[self.write_ptr:end_ptr] = audio_frame
            else:
                # Wraparound case
                first_part_size = self.buffer_size - self.write_ptr
                self.audio_buffer[self.write_ptr:] = audio_frame[:first_part_size]
                self.audio_buffer[:end_ptr] = audio_frame[first_part_size:]
            
            self.write_ptr = end_ptr
            self.total_frames_added += 1
            
            # Mark buffer as full after first complete fill
            if not self.is_full and self.total_frames_added * self.hop_length >= self.buffer_size:
                self.is_full = True
                logger.info(f"Mel audio buffer filled after {self.total_frames_added} frames")
        
        return True
    
    def get_current_audio(self) -> Optional[np.ndarray]:
        """
        Get current audio window for mel-spectrogram processing.
        
        Returns:
            Audio window of context_window seconds, or None if buffer not ready
        """
        with self._lock:
            if not self.is_full:
                return None
            
            # Extract audio in chronological order
            audio_window = np.zeros(self.buffer_size, dtype=np.float32)
            
            if self.write_ptr == 0:
                # No wraparound needed
                audio_window = self.audio_buffer.copy()
            else:
                # Reconstruct chronological order
                audio_window[:self.buffer_size - self.write_ptr] = self.audio_buffer[self.write_ptr:]
                audio_window[self.buffer_size - self.write_ptr:] = self.audio_buffer[:self.write_ptr]
            
            return audio_window
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "context_window": self.context_window,
                "buffer_size": self.buffer_size,
                "hop_length": self.hop_length,
                "total_frames_added": self.total_frames_added,
                "buffer_overruns": self.buffer_overruns,
                "is_full": self.is_full,
                "write_ptr": self.write_ptr,
                "buffer_utilization": self.total_frames_added * self.hop_length / self.buffer_size if self.total_frames_added > 0 else 0.0,
            }


class MelSlidingWindowExtractor:
    """
    Real-time mel-spectrogram extractor with sliding window processing.
    
    Provides efficient frame-by-frame mel-spectrogram extraction with configurable
    context windows and update intervals.
    """
    
    def __init__(
        self,
        context_window: float = 8.5,  # seconds
        update_interval: float = 0.0333,  # 33.3ms
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: Optional[int] = None,  # Will be calculated from update_interval
        win_length: Optional[int] = None,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
        device: str = "cpu",
    ):
        """
        Initialize mel sliding window extractor.
        
        Args:
            context_window: Audio context window in seconds
            update_interval: Update interval in seconds (33.3ms for 30fps)
            sample_rate: Audio sample rate
            n_mels: Number of mel filter banks
            n_fft: FFT window size
            hop_length: STFT hop length (auto-calculated if None)
            win_length: STFT window length (defaults to n_fft)
            f_min: Minimum frequency
            f_max: Maximum frequency (defaults to sr/2)
            power: Power for magnitude spectrogram
            center: Whether to center frames
            pad_mode: Padding mode
            device: Device for processing
        """
        self.context_window = context_window
        self.update_interval = update_interval
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.device = device
        
        # Unified hop_length calculation: int(sample_rate / target_fps)
        target_fps = 1.0 / update_interval  # 30 FPS
        self.hop_length = hop_length or int(sample_rate / target_fps)  # 533 samples
        self.win_length = win_length or n_fft
        
        # Initialize audio buffer
        self.audio_buffer = MelAudioBuffer(
            context_window=context_window,
            sample_rate=sample_rate,
            update_interval=update_interval,
        )
        
        # Create mel-spectrogram transform
        self.mel_transform = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=self.f_max,
        )
        
        # Cache for current features
        self.current_features = None
        self.last_update_time = 0
        self.features_ready = False
        
        # Performance tracking
        self.extraction_times = deque(maxlen=100)
        self.total_extractions = 0
        self.failed_extractions = 0
        
        # Feature dimensions
        expected_frames = int(context_window / update_interval)
        self.feature_shape = (expected_frames, n_mels)
        
        logger.info(f"Mel sliding window extractor initialized:")
        logger.info(f"  Context window: {context_window}s")
        logger.info(f"  Update interval: {update_interval*1000:.1f}ms")
        logger.info(f"  Expected output shape: {self.feature_shape}")
        logger.info(f"  Mel parameters: {n_mels} mels, {n_fft} FFT, {self.hop_length} hop")
    
    def process_audio_frame(self, audio_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process single audio frame and return mel features if ready.
        
        Args:
            audio_frame: Audio frame (should be hop_length samples)
            
        Returns:
            Mel-spectrogram features (T, n_mels) or None if not ready
        """
        # Add frame to buffer
        if not self.audio_buffer.add_audio_frame(audio_frame):
            return None
        
        # Check if we should update features (more aggressive for real-time)
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval * 0.3:  # 30% of interval (very responsive)
            return self.current_features
        
        # Extract features from current buffer
        audio_window = self.audio_buffer.get_current_audio()
        if audio_window is None:
            return None
        
        try:
            start_time = time.time()
            
            # Extract mel-spectrogram using librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio_window,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                power=self.power,
                center=self.center,
                pad_mode=self.pad_mode,
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (T, n_mels) format
            log_mel = log_mel.T  # Now (T, n_mels)
            
            # Ensure consistent output size
            expected_frames = int(self.context_window / self.update_interval)
            if log_mel.shape[0] > expected_frames:
                log_mel = log_mel[:expected_frames]
            elif log_mel.shape[0] < expected_frames:
                # Pad with last frame
                padding = np.tile(log_mel[-1:], (expected_frames - log_mel.shape[0], 1))
                log_mel = np.vstack([log_mel, padding])
            
            # Update cache
            self.current_features = log_mel.astype(np.float32)
            self.last_update_time = current_time
            self.features_ready = True
            
            # Track performance
            extraction_time = time.time() - start_time
            self.extraction_times.append(extraction_time)
            self.total_extractions += 1
            
            return self.current_features
            
        except Exception as e:
            logger.error(f"Mel extraction failed: {e}")
            self.failed_extractions += 1
            return self.current_features  # Return cached features
    
    def process_audio_batch(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio batch for non-real-time usage.
        
        Args:
            audio: Audio array (samples,)
            
        Returns:
            Mel-spectrogram features (T, n_mels)
        """
        try:
            # Extract mel-spectrogram using librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                power=self.power,
                center=self.center,
                pad_mode=self.pad_mode,
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (T, n_mels) format
            log_mel = log_mel.T
            
            return log_mel.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Batch mel extraction failed: {e}")
            # Return dummy features
            duration = len(audio) / self.sample_rate
            frames = int(duration * self.sample_rate / self.hop_length)
            return np.zeros((frames, self.n_mels), dtype=np.float32)
    
    def get_current_features(self) -> Optional[np.ndarray]:
        """Get current cached mel features."""
        return self.current_features if self.features_ready else None
    
    def reset(self):
        """Reset the extractor state."""
        self.audio_buffer = MelAudioBuffer(
            context_window=self.context_window,
            sample_rate=self.sample_rate,
            update_interval=self.update_interval,
        )
        self.current_features = None
        self.last_update_time = 0
        self.features_ready = False
        logger.info("Mel sliding window extractor reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        buffer_stats = self.audio_buffer.get_stats()
        
        extraction_stats = {
            "total_extractions": self.total_extractions,
            "failed_extractions": self.failed_extractions,
            "success_rate": (self.total_extractions - self.failed_extractions) / max(1, self.total_extractions),
            "features_ready": self.features_ready,
        }
        
        if self.extraction_times:
            extraction_stats.update({
                "avg_extraction_time": np.mean(self.extraction_times),
                "max_extraction_time": np.max(self.extraction_times),
                "min_extraction_time": np.min(self.extraction_times),
            })
        
        return {
            "context_window": self.context_window,
            "update_interval": self.update_interval,
            "feature_shape": self.feature_shape,
            "buffer_stats": buffer_stats,
            "extraction_stats": extraction_stats,
        }
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return self.n_mels


def create_mel_extractor(
    context_window: float = 8.5,
    update_interval: float = 0.0333,
    sample_rate: int = 16000,
    n_mels: int = 80,
    **kwargs
) -> MelSlidingWindowExtractor:
    """
    Factory function to create mel sliding window extractor.
    
    Args:
        context_window: Audio context window in seconds
        update_interval: Update interval in seconds
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        **kwargs: Additional parameters for MelSlidingWindowExtractor
        
    Returns:
        Configured mel extractor
    """
    return MelSlidingWindowExtractor(
        context_window=context_window,
        update_interval=update_interval,
        sample_rate=sample_rate,
        n_mels=n_mels,
        **kwargs
    )