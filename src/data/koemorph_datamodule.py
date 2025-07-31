"""
PyTorch Lightning DataModule for new KoeMorph data format.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch

from .koemorph_dataset import KoeMorphDataset, SequentialKoeMorphDataset

logger = logging.getLogger(__name__)


class KoeMorphDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for KoeMorph with new data format support.
    
    Handles:
    - Automatic train/val/test splitting
    - Sequential and random sampling modes
    - Metadata-based FPS detection
    - Data validation
    """
    
    def __init__(
        self,
        train_data_dir: str,
        val_data_dir: Optional[str] = None,
        test_data_dir: Optional[str] = None,
        data_split: Optional[Dict] = None,
        sample_rate: int = 16000,
        target_fps: int = 30,
        num_blendshapes: int = 52,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        sequential: Optional[Dict] = None,
        augmentation: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()
        
        # Data directories
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir) if val_data_dir else self.train_data_dir
        self.test_data_dir = Path(test_data_dir) if test_data_dir else self.train_data_dir
        
        # Data split configuration
        self.data_split = data_split or {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'seed': 42
        }
        
        # Audio/video parameters
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.num_blendshapes = num_blendshapes
        
        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Sequential dataset parameters
        self.sequential = sequential or {
            'window_frames': 256,
            'stride_frames': 1
        }
        
        # Augmentation settings
        self.augmentation = augmentation or {'enable': False}
        
        # Metadata settings
        self.metadata = metadata or {
            'use_timecode': True,
            'verify_sync': True,
            'auto_detect_fps': True
        }
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        
        # Check if we need to split data
        if self.train_data_dir == self.val_data_dir == self.test_data_dir:
            # All data in one directory - need to split
            logger.info("All data in one directory - performing automatic split")
            self._setup_with_split()
        else:
            # Data already split into directories
            logger.info("Using pre-split data directories")
            self._setup_without_split(stage)
    
    def _setup_with_split(self):
        """Set up datasets with automatic train/val/test split."""
        # Load all recordings
        all_recordings = self._get_all_recordings(self.train_data_dir)
        
        # Calculate split sizes
        total = len(all_recordings)
        train_size = int(total * self.data_split['train_ratio'])
        val_size = int(total * self.data_split['val_ratio'])
        test_size = total - train_size - val_size
        
        # Perform split
        generator = torch.Generator().manual_seed(self.data_split['seed'])
        indices = torch.randperm(total, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = SequentialKoeMorphDataset(
            data_dir=self.train_data_dir,
            window_frames=self.sequential['window_frames'],
            stride_frames=self.sequential['stride_frames'],
            sample_rate=self.sample_rate,
            target_fps=self.target_fps,
            shuffle_files=True,
            loop_dataset=True,
        )
        
        # For validation/test, use regular dataset (not sequential)
        self.val_dataset = self._create_subset_dataset(
            self.train_data_dir, 
            [all_recordings[i] for i in val_indices]
        )
        
        self.test_dataset = self._create_subset_dataset(
            self.train_data_dir,
            [all_recordings[i] for i in test_indices]
        )
        
        logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def _setup_without_split(self, stage: Optional[str] = None):
        """Set up datasets from pre-split directories."""
        if stage == "fit" or stage is None:
            # Training dataset (sequential)
            self.train_dataset = SequentialKoeMorphDataset(
                data_dir=self.train_data_dir,
                window_frames=self.sequential['window_frames'],
                stride_frames=self.sequential['stride_frames'],
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                shuffle_files=True,
                loop_dataset=True,
            )
            
            # Validation dataset (regular)
            self.val_dataset = KoeMorphDataset(
                data_dir=self.val_data_dir,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = KoeMorphDataset(
                data_dir=self.test_data_dir,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
            )
    
    def _get_all_recordings(self, data_dir: Path) -> List[str]:
        """Get list of all recording IDs in directory."""
        recordings = []
        for recording_dir in sorted(data_dir.iterdir()):
            if recording_dir.is_dir() and (recording_dir / "metadata.json").exists():
                recordings.append(recording_dir.name)
        return recordings
    
    def _create_subset_dataset(self, data_dir: Path, recording_ids: List[str]) -> KoeMorphDataset:
        """Create a dataset with a subset of recordings."""
        # For now, create full dataset and filter in __getitem__
        # TODO: Implement proper subset filtering
        return KoeMorphDataset(
            data_dir=data_dir,
            sample_rate=self.sample_rate,
            target_fps=self.target_fps,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        # Sequential dataset needs batch_size=1 for collation
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Sequential dataset handles batching internally
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for variable-length sequences."""
        # Find max lengths
        max_audio_len = max(sample['audio'].shape[0] for sample in batch)
        max_blend_len = max(sample['blendshapes'].shape[0] for sample in batch)
        
        # Pad sequences
        audio_batch = []
        blendshapes_batch = []
        masks = []
        
        for sample in batch:
            # Pad audio
            audio = sample['audio']
            audio_pad = torch.zeros(max_audio_len)
            audio_pad[:audio.shape[0]] = audio
            audio_batch.append(audio_pad)
            
            # Pad blendshapes
            blendshapes = sample['blendshapes']
            blendshapes_pad = torch.zeros(max_blend_len, self.num_blendshapes)
            blendshapes_pad[:blendshapes.shape[0]] = blendshapes
            blendshapes_batch.append(blendshapes_pad)
            
            # Create mask
            mask = torch.zeros(max_blend_len, dtype=torch.bool)
            mask[:blendshapes.shape[0]] = True
            masks.append(mask)
        
        return {
            'audio': torch.stack(audio_batch),
            'blendshapes': torch.stack(blendshapes_batch),
            'mask': torch.stack(masks),
            'recording_ids': [s['recording_id'] for s in batch],
        }