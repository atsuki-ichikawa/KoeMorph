"""
Sequential dataset for time-series aware training.

This dataset ensures that audio and blendshape sequences are processed
in temporal order, maintaining continuity for temporal smoothing and
emotion feature consistency.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class SequentialKoeMorphDataset(IterableDataset):
    """
    Sequential dataset that processes audio and blendshapes in temporal order.
    
    This dataset yields overlapping windows of fixed length (256 frames) 
    with configurable stride, maintaining temporal continuity.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        window_frames: int = 256,  # ~8.5 seconds at 30fps
        stride_frames: int = 1,  # 1 frame stride for true sequential learning
        sample_rate: int = 16000,
        target_fps: int = 30,
        shuffle_files: bool = True,
        loop_dataset: bool = True,
        max_files: Optional[int] = None,
    ):
        """
        Initialize sequential dataset.
        
        Args:
            data_dir: Directory containing audio and JSONL files
            window_frames: Number of frames per window (256 = ~8.5s)
            stride_frames: Number of frames to stride between windows
            sample_rate: Audio sample rate
            target_fps: Target frame rate for blendshapes
            shuffle_files: Whether to shuffle file order
            loop_dataset: Whether to loop through dataset indefinitely
            max_files: Maximum number of files to use (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.shuffle_files = shuffle_files
        self.loop_dataset = loop_dataset
        
        # Calculate audio parameters
        self.hop_length = int(sample_rate / target_fps)  # 533 samples
        self.window_samples = window_frames * self.hop_length
        self.stride_samples = stride_frames * self.hop_length
        
        # Find all valid audio/JSONL pairs
        self.file_pairs = self._find_file_pairs()
        
        if max_files:
            self.file_pairs = self.file_pairs[:max_files]
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"No valid audio/JSONL pairs found in {data_dir}")
        
        logger.info(f"Sequential dataset initialized:")
        logger.info(f"  Files: {len(self.file_pairs)}")
        logger.info(f"  Window: {window_frames} frames (~{window_frames/target_fps:.1f}s)")
        logger.info(f"  Stride: {stride_frames} frames (~{stride_frames/target_fps:.1f}s)")
        logger.info(f"  Overlap: {(window_frames-stride_frames)/window_frames*100:.1f}%")
    
    def _find_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching audio and JSONL file pairs."""
        pairs = []
        
        # Look for WAV files
        for audio_path in self.data_dir.glob("**/*.wav"):
            # Find corresponding JSONL
            jsonl_path = audio_path.with_suffix(".jsonl")
            if jsonl_path.exists():
                pairs.append((audio_path, jsonl_path))
        
        return sorted(pairs)  # Consistent ordering
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file."""
        try:
            # Try soundfile first (faster)
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        except Exception:
            # Fallback to librosa
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        return audio.astype(np.float32)
    
    def _load_blendshapes(self, jsonl_path: Path) -> Tuple[np.ndarray, float]:
        """Load blendshape data from JSONL file and detect frame rate."""
        blendshapes = []
        timestamps = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                frame_data = json.loads(line.strip())
                blendshapes.append(frame_data['blendshapes'])
                if 'timestamp' in frame_data:
                    timestamps.append(frame_data['timestamp'])
        
        blendshapes = np.array(blendshapes, dtype=np.float32)
        
        # Detect source frame rate from timestamps
        source_fps = 30.0  # Default assumption
        if len(timestamps) > 1:
            avg_delta = np.mean(np.diff(timestamps))
            if avg_delta > 0:
                source_fps = 1.0 / avg_delta
                # Round to nearest standard frame rate
                if abs(source_fps - 30) < 2:
                    source_fps = 30.0
                elif abs(source_fps - 60) < 2:
                    source_fps = 60.0
        
        return blendshapes, source_fps
    
    def _resample_blendshapes(self, blendshapes: np.ndarray, source_fps: float) -> np.ndarray:
        """Resample blendshapes from source_fps to target_fps."""
        if abs(source_fps - self.target_fps) < 0.1:
            return blendshapes
        
        # Calculate resampling ratio
        ratio = self.target_fps / source_fps
        source_len = len(blendshapes)
        target_len = int(source_len * ratio)
        
        # Create interpolation indices
        source_indices = np.linspace(0, source_len - 1, target_len)
        
        # Interpolate each blendshape dimension
        resampled = np.zeros((target_len, blendshapes.shape[1]), dtype=np.float32)
        for i in range(blendshapes.shape[1]):
            resampled[:, i] = np.interp(source_indices, np.arange(source_len), blendshapes[:, i])
        
        return resampled
    
    def _process_file_pair(self, audio_path: Path, jsonl_path: Path):
        """Process a single audio/JSONL pair yielding sequential windows."""
        try:
            # Load data
            audio = self._load_audio(audio_path)
            blendshapes, source_fps = self._load_blendshapes(jsonl_path)
            
            # Resample blendshapes if needed
            if abs(source_fps - self.target_fps) > 0.1:
                logger.info(f"Resampling blendshapes from {source_fps}fps to {self.target_fps}fps for {audio_path.name}")
                blendshapes = self._resample_blendshapes(blendshapes, source_fps)
            
            # Validate alignment
            expected_frames = len(audio) // self.hop_length
            if abs(len(blendshapes) - expected_frames) > 1:
                logger.warning(
                    f"Frame mismatch in {audio_path.name}: "
                    f"audio suggests {expected_frames} frames, "
                    f"found {len(blendshapes)} blendshapes"
                )
                # Use minimum to ensure alignment
                num_frames = min(len(blendshapes), expected_frames)
                audio = audio[:num_frames * self.hop_length]
                blendshapes = blendshapes[:num_frames]
            
            # Generate sequential windows
            num_windows = (len(blendshapes) - self.window_frames) // self.stride_frames + 1
            
            for i in range(num_windows):
                # Frame indices
                start_frame = i * self.stride_frames
                end_frame = start_frame + self.window_frames
                
                # Audio samples
                start_sample = start_frame * self.hop_length
                end_sample = end_frame * self.hop_length
                
                # Extract windows
                audio_window = audio[start_sample:end_sample]
                blendshape_window = blendshapes[start_frame:end_frame]
                
                # Validate window sizes
                if len(audio_window) == self.window_samples and len(blendshape_window) == self.window_frames:
                    yield {
                        'audio': torch.from_numpy(audio_window),
                        'blendshapes': torch.from_numpy(blendshape_window),
                        'file_indices': torch.tensor(self.file_pairs.index((audio_path, jsonl_path))),
                        'window_indices': torch.tensor(i),
                        'start_frames': torch.tensor(start_frame),
                        'file_names': audio_path.stem,
                    }
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
    
    def __iter__(self):
        """Iterate through dataset yielding sequential windows."""
        while True:  # Loop infinitely if loop_dataset=True
            # Optionally shuffle file order
            file_pairs = self.file_pairs.copy()
            if self.shuffle_files:
                indices = torch.randperm(len(file_pairs)).tolist()
                file_pairs = [file_pairs[i] for i in indices]
            
            # Process each file sequentially
            for audio_path, jsonl_path in file_pairs:
                yield from self._process_file_pair(audio_path, jsonl_path)
            
            if not self.loop_dataset:
                break
    
    def get_num_windows(self) -> int:
        """
        Calculate total number of windows in dataset.
        Note: This requires scanning all files and may be slow.
        """
        total_windows = 0
        
        for audio_path, jsonl_path in self.file_pairs:
            try:
                # Quick estimation based on file size
                audio_info = sf.info(audio_path)
                num_frames = int(audio_info.frames / self.hop_length)
                num_windows = max(0, (num_frames - self.window_frames) // self.stride_frames + 1)
                total_windows += num_windows
            except Exception:
                pass
        
        return total_windows


class SequentialBatchSampler:
    """
    Custom batch sampler that ensures temporal continuity within sequences.
    
    Groups consecutive windows from the same file into batches when possible.
    """
    
    def __init__(
        self,
        dataset: SequentialKoeMorphDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle_sequences: bool = True,
    ):
        """
        Initialize batch sampler.
        
        Args:
            dataset: Sequential dataset
            batch_size: Batch size
            drop_last: Whether to drop incomplete batches
            shuffle_sequences: Whether to shuffle sequence order
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_sequences = shuffle_sequences
    
    def __iter__(self):
        """
        Yield batches maintaining temporal continuity.
        
        Windows from the same file are grouped together when possible.
        """
        batch = []
        current_file_idx = None
        
        for sample in self.dataset:
            # Check if we're still in the same file
            if current_file_idx != sample['file_idx']:
                # New file - yield accumulated batch if any
                if len(batch) >= self.batch_size:
                    yield self._prepare_batch(batch[:self.batch_size])
                    batch = batch[self.batch_size:]
                elif not self.drop_last and batch:
                    yield self._prepare_batch(batch)
                    batch = []
                
                current_file_idx = sample['file_idx']
            
            batch.append(sample)
            
            # Yield full batches
            if len(batch) >= self.batch_size:
                yield self._prepare_batch(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        
        # Handle remaining samples
        if not self.drop_last and batch:
            yield self._prepare_batch(batch)
    
    def _prepare_batch(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare batch dictionary from samples."""
        batch = {
            'audio': torch.stack([s['audio'] for s in samples]),
            'blendshapes': torch.stack([s['blendshapes'] for s in samples]),
            'file_indices': torch.tensor([s['file_idx'] for s in samples]),
            'window_indices': torch.tensor([s['window_idx'] for s in samples]),
            'start_frames': torch.tensor([s['start_frame'] for s in samples]),
        }
        
        # Add file names as metadata
        batch['file_names'] = [s['file_name'] for s in samples]
        
        return batch


def create_sequential_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    window_frames: int = 256,
    stride_frames: int = 128,
    num_workers: int = 2,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a sequential dataloader for time-series aware training.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        window_frames: Window size in frames
        stride_frames: Stride between windows
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader configured for sequential processing
    """
    dataset = SequentialKoeMorphDataset(
        data_dir=data_dir,
        window_frames=window_frames,
        stride_frames=stride_frames,
        **kwargs
    )
    
    # For IterableDataset, we use regular DataLoader with batch_size
    # The sequential ordering is maintained by the dataset itself
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    
    return dataloader