"""
Data I/O module for ARKit blendshape and audio data.

Unit name     : ARKit Data Loader
Input         : jsonl file path + wav file path  
Output        : {'wav': torch.FloatTensor (L,), 'arkit': torch.FloatTensor (T, 52)}
Dependencies  : soundfile, json
Assumptions   : Synchronized timestamps, 52 blendshapes
Failure modes : Missing files, timestamp mismatch, invalid blendshape count
Test cases    : test_load_sample, test_sync_validation, test_blendshape_count
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch


class ARKitDataLoader:
    """Loader for synchronized ARKit blendshape and audio data."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        max_time_drift: float = 0.1,
    ):
        """
        Initialize the ARKit data loader.
        
        Args:
            sample_rate: Target audio sample rate (Hz)
            target_fps: Target blendshape frame rate (FPS)
            max_time_drift: Maximum allowed time drift between audio/blendshapes (seconds)
        """
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.max_time_drift = max_time_drift
    
    def load_sample(
        self, 
        jsonl_path: Union[str, Path], 
        wav_path: Union[str, Path]
    ) -> Dict[str, torch.Tensor]:
        """
        Load a single sample with synchronized audio and blendshapes.
        
        Args:
            jsonl_path: Path to jsonl file with timestamped blendshapes
            wav_path: Path to wav audio file
            
        Returns:
            Dictionary with 'wav' (L,) and 'arkit' (T, 52) tensors
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If data format is invalid or sync is poor
        """
        jsonl_path = Path(jsonl_path)
        wav_path = Path(wav_path)
        
        # Validate file existence
        if not jsonl_path.exists():
            raise FileNotFoundError(f"ARKit file not found: {jsonl_path}")
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        # Load audio
        audio, audio_sr = sf.read(str(wav_path), dtype='float32')
        if audio_sr != self.sample_rate:
            warnings.warn(
                f"Audio sample rate {audio_sr} != target {self.sample_rate}. "
                "Consider resampling for best results."
            )
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Load blendshapes
        blendshapes = self._load_blendshapes(jsonl_path)
        
        # Validate synchronization
        audio_duration = len(audio) / audio_sr
        bs_duration = len(blendshapes) / self.target_fps
        time_drift = abs(audio_duration - bs_duration)
        
        if time_drift > self.max_time_drift:
            raise ValueError(
                f"Time drift {time_drift:.3f}s exceeds threshold {self.max_time_drift}s. "
                f"Audio: {audio_duration:.3f}s, Blendshapes: {bs_duration:.3f}s"
            )
        
        return {
            'wav': torch.from_numpy(audio),
            'arkit': torch.from_numpy(blendshapes)
        }
    
    def _load_blendshapes(self, jsonl_path: Path) -> np.ndarray:
        """
        Load blendshapes from jsonl file.
        
        Args:
            jsonl_path: Path to jsonl file
            
        Returns:
            Numpy array of shape (T, 52) with blendshape coefficients
        """
        blendshapes = []
        timestamps = []
        
        with open(jsonl_path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_no}: {e}")
                
                # Extract timestamp and blendshapes
                if 'timestamp' not in data:
                    raise ValueError(f"Missing 'timestamp' field at line {line_no}")
                if 'blendshapes' not in data:
                    raise ValueError(f"Missing 'blendshapes' field at line {line_no}")
                
                timestamp = data['timestamp']
                bs_values = data['blendshapes']
                
                # Validate blendshape count
                if len(bs_values) != 52:
                    raise ValueError(
                        f"Expected 52 blendshapes, got {len(bs_values)} at line {line_no}"
                    )
                
                # Validate blendshape range [0, 1]
                bs_array = np.array(bs_values, dtype=np.float32)
                if not np.all((bs_array >= 0) & (bs_array <= 1)):
                    warnings.warn(f"Blendshape values outside [0,1] range at line {line_no}")
                
                timestamps.append(timestamp)
                blendshapes.append(bs_array)
        
        if not blendshapes:
            raise ValueError("No blendshape data found in file")
        
        # Convert to numpy array and validate temporal consistency
        blendshapes = np.stack(blendshapes, axis=0)  # (T, 52)
        timestamps = np.array(timestamps)
        
        # Check temporal ordering
        if not np.all(np.diff(timestamps) > 0):
            warnings.warn("Non-monotonic timestamps detected")
        
        return blendshapes
    
    def load_batch(
        self, 
        file_pairs: List[Tuple[Union[str, Path], Union[str, Path]]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Load multiple samples as a batch.
        
        Args:
            file_pairs: List of (jsonl_path, wav_path) tuples
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        for jsonl_path, wav_path in file_pairs:
            try:
                sample = self.load_sample(jsonl_path, wav_path)
                samples.append(sample)
            except (FileNotFoundError, ValueError) as e:
                warnings.warn(f"Failed to load {jsonl_path}, {wav_path}: {e}")
                continue
        
        return samples


def validate_data_consistency(
    samples: List[Dict[str, torch.Tensor]], 
    tolerance: float = 0.05
) -> Dict[str, bool]:
    """
    Validate consistency across multiple samples.
    
    Args:
        samples: List of loaded samples
        tolerance: Relative tolerance for duration checks
        
    Returns:
        Dictionary with validation results
    """
    if not samples:
        return {'valid': False, 'reason': 'No samples provided'}
    
    results = {'valid': True, 'issues': []}
    
    # Check blendshape dimension consistency
    bs_shapes = [s['arkit'].shape for s in samples]
    if not all(shape[1] == 52 for shape in bs_shapes):
        results['valid'] = False
        results['issues'].append('Inconsistent blendshape dimensions')
    
    # Check audio/blendshape duration consistency within samples
    for i, sample in enumerate(samples):
        audio_duration = len(sample['wav']) / 16000  # Assuming 16kHz
        bs_duration = len(sample['arkit']) / 30.0    # Assuming 30 FPS
        
        relative_error = abs(audio_duration - bs_duration) / max(audio_duration, bs_duration)
        if relative_error > tolerance:
            results['valid'] = False
            results['issues'].append(
                f'Sample {i}: duration mismatch {relative_error:.3f} > {tolerance}'
            )
    
    return results