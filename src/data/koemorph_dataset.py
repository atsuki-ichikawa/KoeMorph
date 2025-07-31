"""
KoeMorph Dataset for new data format with timecode-based synchronization.

This dataset supports the new folder structure and metadata format
introduced in koemorph-data-converter v2.0.
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


class KoeMorphDataset(Dataset):
    """
    Dataset for KoeMorph with new data format.
    
    Supports:
    - New folder structure (each recording in separate directory)
    - Timecode-based synchronization
    - Metadata files (metadata.json, timecode.json)
    - Automatic FPS detection and resampling
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 16000,
        target_fps: int = 30,
        max_length: Optional[float] = None,
        transform=None,
    ):
        """
        Initialize KoeMorph dataset.
        
        Args:
            data_dir: Directory containing processed recordings
            sample_rate: Target audio sample rate
            target_fps: Target frame rate for blendshapes
            max_length: Maximum length in seconds (optional)
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.max_length = max_length
        self.transform = transform
        
        # Load all recordings
        self.recordings = self._load_recordings()
        
        if len(self.recordings) == 0:
            raise ValueError(f"No valid recordings found in {data_dir}")
        
        logger.info(f"Loaded {len(self.recordings)} recordings")
        logger.info(f"Target FPS: {target_fps}, Sample rate: {sample_rate}")
    
    def _load_recordings(self) -> List[Dict]:
        """Load all recordings from data directory."""
        recordings = []
        
        # Iterate through all subdirectories
        for recording_dir in sorted(self.data_dir.iterdir()):
            if not recording_dir.is_dir():
                continue
            
            # Check if all required files exist
            audio_path = recording_dir / "audio.wav"
            blendshapes_path = recording_dir / "blendshapes.jsonl"
            metadata_path = recording_dir / "metadata.json"
            timecode_path = recording_dir / "timecode.json"
            
            if not all(p.exists() for p in [audio_path, blendshapes_path, metadata_path]):
                logger.warning(f"Skipping incomplete recording: {recording_dir.name}")
                continue
            
            try:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load timecode info if available
                timecode_info = None
                if timecode_path.exists():
                    with open(timecode_path, 'r') as f:
                        timecode_info = json.load(f)
                
                recordings.append({
                    'id': metadata['recording_id'],
                    'dir': recording_dir,
                    'audio_path': audio_path,
                    'blendshapes_path': blendshapes_path,
                    'metadata': metadata,
                    'timecode_info': timecode_info,
                    'source_fps': metadata.get('output_fps', 60),  # FPS of the data
                })
                
            except Exception as e:
                logger.error(f"Error loading recording {recording_dir.name}: {e}")
                continue
        
        return recordings
    
    def __len__(self) -> int:
        return len(self.recordings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single recording."""
        recording = self.recordings[idx]
        
        # Load audio
        audio = self._load_audio(recording['audio_path'])
        
        # Load blendshapes
        blendshapes, timestamps = self._load_blendshapes(
            recording['blendshapes_path'],
            recording['source_fps']
        )
        
        # Apply max length if specified
        if self.max_length is not None:
            max_samples = int(self.max_length * self.sample_rate)
            max_frames = int(self.max_length * self.target_fps)
            
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            if len(blendshapes) > max_frames:
                blendshapes = blendshapes[:max_frames]
                timestamps = timestamps[:max_frames]
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio).float()
        blendshapes_tensor = torch.from_numpy(blendshapes).float()
        
        # Create sample
        sample = {
            'audio': audio_tensor,
            'blendshapes': blendshapes_tensor,
            'timestamps': timestamps,
            'recording_id': recording['id'],
            'source_fps': recording['source_fps'],
            'target_fps': self.target_fps,
        }
        
        # Add metadata if needed
        if recording['metadata']:
            sample['metadata'] = recording['metadata']
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load and resample audio if needed."""
        try:
            # Try soundfile first (faster)
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        except Exception:
            # Fallback to librosa
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        return audio.astype(np.float32)
    
    def _load_blendshapes(self, blendshapes_path: Path, source_fps: int) -> Tuple[np.ndarray, List[float]]:
        """Load blendshapes and resample if needed."""
        frames = []
        timestamps = []
        
        # Read JSONL file
        with open(blendshapes_path, 'r') as f:
            for line in f:
                frame_data = json.loads(line.strip())
                frames.append(frame_data['blendshapes'])
                timestamps.append(frame_data['timestamp'])
        
        blendshapes = np.array(frames, dtype=np.float32)
        
        # Resample if needed
        if source_fps != self.target_fps:
            blendshapes, timestamps = self._resample_blendshapes(
                blendshapes, timestamps, source_fps, self.target_fps
            )
        
        return blendshapes, timestamps
    
    def _resample_blendshapes(
        self, 
        blendshapes: np.ndarray, 
        timestamps: List[float],
        source_fps: int, 
        target_fps: int
    ) -> Tuple[np.ndarray, List[float]]:
        """Resample blendshapes to target FPS."""
        if abs(source_fps - target_fps) < 0.1:
            return blendshapes, timestamps
        
        # Calculate target timestamps
        duration = timestamps[-1]
        target_timestamps = np.arange(0, duration, 1.0 / target_fps)
        
        # Interpolate each blendshape dimension
        resampled = np.zeros((len(target_timestamps), blendshapes.shape[1]), dtype=np.float32)
        for i in range(blendshapes.shape[1]):
            resampled[:, i] = np.interp(target_timestamps, timestamps, blendshapes[:, i])
        
        logger.debug(f"Resampled from {source_fps}fps to {target_fps}fps: {len(blendshapes)} -> {len(resampled)} frames")
        
        return resampled, target_timestamps.tolist()


class SequentialKoeMorphDataset(IterableDataset):
    """
    Sequential dataset for KoeMorph with sliding window support.
    
    This dataset yields overlapping windows for sequential training,
    compatible with the new data format.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        window_frames: int = 256,
        stride_frames: int = 1,
        sample_rate: int = 16000,
        target_fps: int = 30,
        shuffle_files: bool = True,
        loop_dataset: bool = True,
        max_files: Optional[int] = None,
    ):
        """
        Initialize sequential dataset.
        
        Args:
            data_dir: Directory containing processed recordings
            window_frames: Number of frames per window
            stride_frames: Number of frames to stride between windows
            sample_rate: Audio sample rate
            target_fps: Target frame rate for blendshapes
            shuffle_files: Whether to shuffle file order
            loop_dataset: Whether to loop through dataset indefinitely
            max_files: Maximum number of files to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.shuffle_files = shuffle_files
        self.loop_dataset = loop_dataset
        
        # Calculate audio parameters
        self.hop_length = int(sample_rate / target_fps)
        self.window_samples = window_frames * self.hop_length
        self.stride_samples = stride_frames * self.hop_length
        
        # Load recordings
        self.recordings = self._load_recordings()
        
        if max_files:
            self.recordings = self.recordings[:max_files]
        
        logger.info(f"Sequential dataset initialized:")
        logger.info(f"  Recordings: {len(self.recordings)}")
        logger.info(f"  Window: {window_frames} frames (~{window_frames/target_fps:.1f}s)")
        logger.info(f"  Stride: {stride_frames} frames (~{stride_frames/target_fps:.1f}s)")
        logger.info(f"  Target FPS: {target_fps}")
    
    def _load_recordings(self) -> List[Dict]:
        """Load recording metadata."""
        recordings = []
        
        for recording_dir in sorted(self.data_dir.iterdir()):
            if not recording_dir.is_dir():
                continue
            
            # Check required files
            audio_path = recording_dir / "audio.wav"
            blendshapes_path = recording_dir / "blendshapes.jsonl"
            metadata_path = recording_dir / "metadata.json"
            
            if not all(p.exists() for p in [audio_path, blendshapes_path, metadata_path]):
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                recordings.append({
                    'id': metadata['recording_id'],
                    'audio_path': audio_path,
                    'blendshapes_path': blendshapes_path,
                    'metadata': metadata,
                    'source_fps': metadata.get('output_fps', 60),
                })
            except Exception as e:
                logger.error(f"Error loading {recording_dir.name}: {e}")
        
        return recordings
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file."""
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        except Exception:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        return audio.astype(np.float32)
    
    def _load_blendshapes(self, blendshapes_path: Path, source_fps: int) -> np.ndarray:
        """Load and resample blendshapes."""
        frames = []
        timestamps = []
        
        with open(blendshapes_path, 'r') as f:
            for line in f:
                frame_data = json.loads(line.strip())
                frames.append(frame_data['blendshapes'])
                timestamps.append(frame_data['timestamp'])
        
        blendshapes = np.array(frames, dtype=np.float32)
        
        # Resample if needed
        if abs(source_fps - self.target_fps) > 0.1:
            duration = timestamps[-1]
            target_timestamps = np.arange(0, duration, 1.0 / self.target_fps)
            
            resampled = np.zeros((len(target_timestamps), blendshapes.shape[1]), dtype=np.float32)
            for i in range(blendshapes.shape[1]):
                resampled[:, i] = np.interp(target_timestamps, timestamps, blendshapes[:, i])
            
            blendshapes = resampled
        
        return blendshapes
    
    def _process_recording(self, recording: Dict):
        """Process a single recording yielding sequential windows."""
        try:
            # Load data
            audio = self._load_audio(recording['audio_path'])
            blendshapes = self._load_blendshapes(
                recording['blendshapes_path'],
                recording['source_fps']
            )
            
            # Validate alignment
            expected_frames = len(audio) // self.hop_length
            if abs(len(blendshapes) - expected_frames) > 1:
                logger.warning(
                    f"Frame mismatch in {recording['id']}: "
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
                
                # Audio indices
                start_sample = start_frame * self.hop_length
                end_sample = end_frame * self.hop_length
                
                # Extract windows
                audio_window = audio[start_sample:end_sample]
                blendshapes_window = blendshapes[start_frame:end_frame]
                
                yield {
                    'audio': torch.from_numpy(audio_window).float(),
                    'blendshapes': torch.from_numpy(blendshapes_window).float(),
                    'recording_id': recording['id'],
                    'window_idx': i,
                    'start_frame': start_frame,
                }
        
        except Exception as e:
            logger.error(f"Error processing {recording['id']}: {e}")
    
    def __iter__(self):
        """Iterate through dataset yielding sequential windows."""
        while True:
            # Optionally shuffle recordings
            recordings = self.recordings.copy()
            if self.shuffle_files:
                indices = torch.randperm(len(recordings)).tolist()
                recordings = [recordings[i] for i in indices]
            
            # Process each recording
            for recording in recordings:
                yield from self._process_recording(recording)
            
            if not self.loop_dataset:
                break