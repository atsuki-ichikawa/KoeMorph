"""
Adaptive sequential dataset with configurable stride strategies.

This dataset supports both dense (1-frame) and sparse sampling strategies
for efficient training while maintaining temporal consistency.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class AdaptiveSequentialDataset(IterableDataset):
    """
    Adaptive sequential dataset with multiple stride strategies.
    
    Supports:
    1. Dense mode: 1-frame stride for maximum temporal learning
    2. Progressive mode: Gradually decrease stride during training
    3. Mixed mode: Alternate between dense and sparse sampling
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        window_frames: int = 256,
        stride_mode: str = "progressive",  # "dense", "sparse", "progressive", "mixed"
        initial_stride: int = 32,
        final_stride: int = 1,
        epoch: int = 0,
        max_epochs: int = 100,
        sample_rate: int = 16000,
        target_fps: int = 30,
        dense_sampling_ratio: float = 0.1,  # For mixed mode
        shuffle_files: bool = True,
        loop_dataset: bool = True,
        max_files: Optional[int] = None,
    ):
        """
        Initialize adaptive sequential dataset.
        
        Args:
            data_dir: Directory containing audio and JSONL files
            window_frames: Number of frames per window (256 = ~8.5s)
            stride_mode: Stride strategy ("dense", "sparse", "progressive", "mixed")
            initial_stride: Starting stride for progressive mode
            final_stride: Target stride for progressive mode
            epoch: Current epoch (for progressive mode)
            max_epochs: Total epochs (for progressive mode)
            sample_rate: Audio sample rate
            target_fps: Target frame rate
            dense_sampling_ratio: Ratio of dense samples in mixed mode
            shuffle_files: Whether to shuffle file order
            loop_dataset: Whether to loop indefinitely
            max_files: Maximum files to use
        """
        self.data_dir = Path(data_dir)
        self.window_frames = window_frames
        self.stride_mode = stride_mode
        self.initial_stride = initial_stride
        self.final_stride = final_stride
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.dense_sampling_ratio = dense_sampling_ratio
        self.shuffle_files = shuffle_files
        self.loop_dataset = loop_dataset
        
        # Calculate audio parameters
        self.hop_length = int(sample_rate / target_fps)  # 533 samples
        self.window_samples = window_frames * self.hop_length
        
        # Find all valid audio/JSONL pairs
        self.file_pairs = self._find_file_pairs()
        
        if max_files:
            self.file_pairs = self.file_pairs[:max_files]
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"No valid audio/JSONL pairs found in {data_dir}")
        
        # Calculate current stride based on mode
        self.current_stride = self._calculate_stride()
        
        logger.info(f"Adaptive sequential dataset initialized:")
        logger.info(f"  Files: {len(self.file_pairs)}")
        logger.info(f"  Window: {window_frames} frames (~{window_frames/target_fps:.1f}s)")
        logger.info(f"  Stride mode: {stride_mode}")
        logger.info(f"  Current stride: {self.current_stride} frames")
    
    def _find_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching audio and JSONL file pairs."""
        pairs = []
        
        for audio_path in self.data_dir.glob("**/*.wav"):
            jsonl_path = audio_path.with_suffix(".jsonl")
            if jsonl_path.exists():
                pairs.append((audio_path, jsonl_path))
        
        return sorted(pairs)
    
    def _calculate_stride(self) -> int:
        """Calculate current stride based on mode and epoch."""
        if self.stride_mode == "dense":
            return 1
        elif self.stride_mode == "sparse":
            return self.initial_stride
        elif self.stride_mode == "progressive":
            # Linearly decrease stride over epochs
            progress = min(1.0, self.epoch / max(1, self.max_epochs - 1))
            stride = int(self.initial_stride - progress * (self.initial_stride - self.final_stride))
            return max(self.final_stride, stride)
        elif self.stride_mode == "mixed":
            # Will be determined per-window
            return self.initial_stride
        else:
            raise ValueError(f"Unknown stride mode: {self.stride_mode}")
    
    def set_epoch(self, epoch: int):
        """Update epoch for progressive stride calculation."""
        self.epoch = epoch
        self.current_stride = self._calculate_stride()
        logger.info(f"Epoch {epoch}: stride updated to {self.current_stride}")
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file."""
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        except Exception:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        return audio.astype(np.float32)
    
    def _load_blendshapes(self, jsonl_path: Path) -> np.ndarray:
        """Load blendshape data from JSONL file."""
        blendshapes = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                frame_data = json.loads(line.strip())
                blendshapes.append(frame_data['blendshapes'])
        
        return np.array(blendshapes, dtype=np.float32)
    
    def _dense_windows(self, audio: np.ndarray, blendshapes: np.ndarray, 
                      audio_path: Path, jsonl_path: Path) -> Iterator[Dict]:
        """Generate windows with 1-frame stride (dense sampling)."""
        num_windows = len(blendshapes) - self.window_frames + 1
        
        for i in range(num_windows):
            start_frame = i
            end_frame = start_frame + self.window_frames
            
            start_sample = start_frame * self.hop_length
            end_sample = end_frame * self.hop_length
            
            audio_window = audio[start_sample:end_sample]
            blendshape_window = blendshapes[start_frame:end_frame]
            
            if len(audio_window) == self.window_samples and len(blendshape_window) == self.window_frames:
                yield {
                    'audio': torch.from_numpy(audio_window),
                    'blendshapes': torch.from_numpy(blendshape_window),
                    'file_indices': torch.tensor(self.file_pairs.index((audio_path, jsonl_path))),
                    'window_indices': torch.tensor(i),
                    'start_frames': torch.tensor(start_frame),
                    'file_names': audio_path.stem,
                    'is_dense': torch.tensor(True),
                }
    
    def _sparse_windows(self, audio: np.ndarray, blendshapes: np.ndarray,
                       audio_path: Path, jsonl_path: Path, stride: int) -> Iterator[Dict]:
        """Generate windows with configurable stride (sparse sampling)."""
        num_windows = (len(blendshapes) - self.window_frames) // stride + 1
        
        for i in range(num_windows):
            start_frame = i * stride
            end_frame = start_frame + self.window_frames
            
            if end_frame > len(blendshapes):
                break
            
            start_sample = start_frame * self.hop_length
            end_sample = end_frame * self.hop_length
            
            audio_window = audio[start_sample:end_sample]
            blendshape_window = blendshapes[start_frame:end_frame]
            
            if len(audio_window) == self.window_samples and len(blendshape_window) == self.window_frames:
                yield {
                    'audio': torch.from_numpy(audio_window),
                    'blendshapes': torch.from_numpy(blendshape_window),
                    'file_indices': torch.tensor(self.file_pairs.index((audio_path, jsonl_path))),
                    'window_indices': torch.tensor(i),
                    'start_frames': torch.tensor(start_frame),
                    'file_names': audio_path.stem,
                    'is_dense': torch.tensor(False),
                }
    
    def _process_file_pair(self, audio_path: Path, jsonl_path: Path) -> Iterator[Dict]:
        """Process a single audio/JSONL pair with adaptive stride."""
        try:
            # Load data
            audio = self._load_audio(audio_path)
            blendshapes = self._load_blendshapes(jsonl_path)
            
            # Validate alignment
            expected_frames = len(audio) // self.hop_length
            if abs(len(blendshapes) - expected_frames) > 1:
                num_frames = min(len(blendshapes), expected_frames)
                audio = audio[:num_frames * self.hop_length]
                blendshapes = blendshapes[:num_frames]
            
            # Generate windows based on mode
            if self.stride_mode == "dense":
                yield from self._dense_windows(audio, blendshapes, audio_path, jsonl_path)
            
            elif self.stride_mode == "sparse" or self.stride_mode == "progressive":
                yield from self._sparse_windows(audio, blendshapes, audio_path, jsonl_path, 
                                              self.current_stride)
            
            elif self.stride_mode == "mixed":
                # Mix dense and sparse sampling
                # First, some dense samples
                dense_samples = int((len(blendshapes) - self.window_frames) * self.dense_sampling_ratio)
                dense_indices = np.random.choice(
                    len(blendshapes) - self.window_frames,
                    size=dense_samples,
                    replace=False
                )
                
                # Generate dense samples
                for idx in sorted(dense_indices):
                    start_frame = idx
                    end_frame = start_frame + self.window_frames
                    
                    start_sample = start_frame * self.hop_length
                    end_sample = end_frame * self.hop_length
                    
                    audio_window = audio[start_sample:end_sample]
                    blendshape_window = blendshapes[start_frame:end_frame]
                    
                    if len(audio_window) == self.window_samples:
                        yield {
                            'audio': torch.from_numpy(audio_window),
                            'blendshapes': torch.from_numpy(blendshape_window),
                            'file_indices': torch.tensor(self.file_pairs.index((audio_path, jsonl_path))),
                            'window_indices': torch.tensor(idx),
                            'start_frames': torch.tensor(start_frame),
                            'file_names': audio_path.stem,
                            'is_dense': torch.tensor(True),
                        }
                
                # Then sparse samples
                yield from self._sparse_windows(audio, blendshapes, audio_path, jsonl_path,
                                              self.initial_stride)
                
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
    
    def __iter__(self):
        """Iterate through dataset with adaptive stride."""
        while True:
            file_pairs = self.file_pairs.copy()
            if self.shuffle_files:
                indices = torch.randperm(len(file_pairs)).tolist()
                file_pairs = [file_pairs[i] for i in indices]
            
            for audio_path, jsonl_path in file_pairs:
                yield from self._process_file_pair(audio_path, jsonl_path)
            
            if not self.loop_dataset:
                break
    
    def estimate_epoch_size(self) -> int:
        """Estimate number of windows per epoch."""
        if self.stride_mode == "dense":
            # Rough estimate for dense mode
            avg_file_frames = 300  # Assume ~10 seconds average
            windows_per_file = max(0, avg_file_frames - self.window_frames + 1)
            return len(self.file_pairs) * windows_per_file
        
        elif self.stride_mode == "sparse" or self.stride_mode == "progressive":
            avg_file_frames = 300
            windows_per_file = max(0, (avg_file_frames - self.window_frames) // self.current_stride + 1)
            return len(self.file_pairs) * windows_per_file
        
        elif self.stride_mode == "mixed":
            avg_file_frames = 300
            dense_windows = int((avg_file_frames - self.window_frames) * self.dense_sampling_ratio)
            sparse_windows = (avg_file_frames - self.window_frames) // self.initial_stride + 1
            return len(self.file_pairs) * (dense_windows + sparse_windows)
        
        return 0


def create_adaptive_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    stride_mode: str = "progressive",
    epoch: int = 0,
    max_epochs: int = 100,
    num_workers: int = 2,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create adaptive sequential dataloader.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        stride_mode: Stride strategy
        epoch: Current epoch
        max_epochs: Total epochs
        num_workers: Data loading workers
        **kwargs: Additional dataset arguments
        
    Returns:
        Configured DataLoader
    """
    dataset = AdaptiveSequentialDataset(
        data_dir=data_dir,
        stride_mode=stride_mode,
        epoch=epoch,
        max_epochs=max_epochs,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    
    return dataloader