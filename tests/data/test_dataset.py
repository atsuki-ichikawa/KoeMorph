"""Tests for data.dataset module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KoeMorphDataModule, KoeMorphDataset, collate_fn


class TestKoeMorphDataset:
    """Test KoeMorphDataset functionality."""
    
    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a directory with sample paired files."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Create 3 sample pairs
        for i in range(3):
            # Audio file
            wav_path = data_dir / f"sample_{i:02d}.wav"
            audio = 0.1 * np.random.randn(8000).astype(np.float32)  # 0.5 seconds at 16kHz
            sf.write(str(wav_path), audio, 16000)
            
            # Blendshape file
            jsonl_path = data_dir / f"sample_{i:02d}.jsonl"
            with open(jsonl_path, 'w') as f:
                for j in range(15):  # 0.5 seconds at 30 FPS
                    data = {
                        'timestamp': j / 30.0,
                        'blendshapes': np.random.rand(52).tolist()
                    }
                    f.write(json.dumps(data) + '\n')
        
        return data_dir
    
    def test_dataset_creation(self, sample_data_dir):
        """Test dataset creation and file discovery."""
        dataset = KoeMorphDataset(sample_data_dir)
        
        assert len(dataset) == 3
        assert len(dataset.file_pairs) == 3
    
    def test_dataset_getitem(self, sample_data_dir):
        """Test dataset __getitem__ method."""
        dataset = KoeMorphDataset(sample_data_dir)
        
        sample = dataset[0]
        
        assert 'wav' in sample
        assert 'arkit' in sample
        assert isinstance(sample['wav'], torch.Tensor)
        assert isinstance(sample['arkit'], torch.Tensor)
        assert sample['wav'].dim() == 1
        assert sample['arkit'].shape[1] == 52
    
    def test_dataset_length_limiting(self, sample_data_dir):
        """Test audio length limiting functionality."""
        dataset = KoeMorphDataset(sample_data_dir, max_audio_length=0.25)  # 0.25 seconds
        
        sample = dataset[0]
        
        max_audio_samples = int(0.25 * 16000)  # 4000 samples
        max_bs_frames = int(0.25 * 30)         # 7.5 -> 7 frames
        
        assert len(sample['wav']) <= max_audio_samples
        assert len(sample['arkit']) <= max_bs_frames
    
    def test_dataset_empty_directory(self, tmp_path):
        """Test error handling for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No paired files found"):
            KoeMorphDataset(empty_dir)
    
    def test_dataset_missing_wav(self, tmp_path):
        """Test handling of missing wav files."""
        data_dir = tmp_path / "incomplete"
        data_dir.mkdir()
        
        # Create jsonl without corresponding wav
        jsonl_path = data_dir / "orphan.jsonl"
        with open(jsonl_path, 'w') as f:
            data = {'timestamp': 0.0, 'blendshapes': [0.5] * 52}
            f.write(json.dumps(data) + '\n')
        
        # Should find no pairs
        with pytest.raises(ValueError, match="No paired files found"):
            KoeMorphDataset(data_dir)
    
    def test_get_sample_info(self, sample_data_dir):
        """Test sample info retrieval."""
        dataset = KoeMorphDataset(sample_data_dir)
        
        info = dataset.get_sample_info(0)
        
        assert 'jsonl_path' in info
        assert 'wav_path' in info
        assert 'basename' in info
        assert info['basename'] == 'sample_00'


class TestCollateFn:
    """Test custom collate function."""
    
    def test_collate_basic(self):
        """Test basic collation functionality."""
        # Create samples with different lengths
        batch = [
            {
                'wav': torch.randn(1000),      # Short audio
                'arkit': torch.rand(10, 52)    # Short blendshapes
            },
            {
                'wav': torch.randn(2000),      # Long audio
                'arkit': torch.rand(20, 52)    # Long blendshapes
            }
        ]
        
        result = collate_fn(batch)
        
        # Check output structure
        expected_keys = {'wav', 'arkit', 'audio_mask', 'blendshape_mask', 
                        'audio_lengths', 'blendshape_lengths'}
        assert set(result.keys()) == expected_keys
        
        # Check shapes
        assert result['wav'].shape == (2, 2000)  # Padded to max length
        assert result['arkit'].shape == (2, 20, 52)  # Padded to max length
        assert result['audio_mask'].shape == (2, 2000)
        assert result['blendshape_mask'].shape == (2, 20)
        
        # Check lengths
        assert result['audio_lengths'].tolist() == [1000, 2000]
        assert result['blendshape_lengths'].tolist() == [10, 20]
    
    def test_collate_masks(self):
        """Test mask generation in collate function."""
        batch = [
            {
                'wav': torch.ones(100),      
                'arkit': torch.ones(5, 52)   
            },
            {
                'wav': torch.ones(200),      
                'arkit': torch.ones(10, 52)  
            }
        ]
        
        result = collate_fn(batch)
        
        # Check audio mask
        audio_mask = result['audio_mask']
        assert audio_mask[0, :100].all()    # First 100 should be True
        assert not audio_mask[0, 100:].any()  # Rest should be False
        assert audio_mask[1].all()           # All should be True for second sample
        
        # Check blendshape mask
        bs_mask = result['blendshape_mask']
        assert bs_mask[0, :5].all()         # First 5 should be True
        assert not bs_mask[0, 5:].any()     # Rest should be False
        assert bs_mask[1].all()             # All should be True for second sample


class TestKoeMorphDataModule:
    """Test KoeMorphDataModule functionality."""
    
    @pytest.fixture
    def data_dirs(self, tmp_path):
        """Create train/val/test directories with sample data."""
        dirs = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = tmp_path / split
            split_dir.mkdir()
            dirs[split] = split_dir
            
            # Create 2 samples per split
            for i in range(2):
                # Audio
                wav_path = split_dir / f"{split}_{i:02d}.wav"
                audio = 0.1 * np.random.randn(4000).astype(np.float32)
                sf.write(str(wav_path), audio, 16000)
                
                # Blendshapes
                jsonl_path = split_dir / f"{split}_{i:02d}.jsonl"
                with open(jsonl_path, 'w') as f:
                    for j in range(8):  # 8 frames
                        data = {
                            'timestamp': j / 30.0,
                            'blendshapes': np.random.rand(52).tolist()
                        }
                        f.write(json.dumps(data) + '\n')
        
        return dirs
    
    def test_datamodule_creation(self, data_dirs):
        """Test data module creation."""
        dm = KoeMorphDataModule(
            train_data_dir=data_dirs['train'],
            val_data_dir=data_dirs['val'],
            test_data_dir=data_dirs['test'],
            batch_size=2,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        # Should not create datasets until setup
        assert dm._train_dataset is None
        assert dm._val_dataset is None
        assert dm._test_dataset is None
    
    def test_datamodule_setup(self, data_dirs):
        """Test data module setup."""
        dm = KoeMorphDataModule(
            train_data_dir=data_dirs['train'],
            val_data_dir=data_dirs['val'],
            test_data_dir=data_dirs['test'],
            batch_size=2,
            num_workers=0
        )
        
        dm.setup()
        
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert dm._test_dataset is not None
        assert len(dm._train_dataset) == 2
        assert len(dm._val_dataset) == 2
        assert len(dm._test_dataset) == 2
    
    def test_datamodule_dataloaders(self, data_dirs):
        """Test DataLoader creation."""
        dm = KoeMorphDataModule(
            train_data_dir=data_dirs['train'],
            val_data_dir=data_dirs['val'],
            batch_size=2,
            num_workers=0
        )
        
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        
        # Test getting a batch
        train_batch = next(iter(train_loader))
        assert 'wav' in train_batch
        assert 'arkit' in train_batch
        assert train_batch['wav'].shape[0] == 2  # Batch size
    
    def test_datamodule_train_only(self, data_dirs):
        """Test data module with only training data."""
        dm = KoeMorphDataModule(
            train_data_dir=data_dirs['train'],
            batch_size=1,
            num_workers=0
        )
        
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        
        assert train_loader is not None
        assert val_loader is None
        assert test_loader is None
    
    def test_datamodule_stats(self, data_dirs):
        """Test dataset statistics."""
        dm = KoeMorphDataModule(
            train_data_dir=data_dirs['train'],
            val_data_dir=data_dirs['val'],
            test_data_dir=data_dirs['test'],
            batch_size=2,
            num_workers=0
        )
        
        stats = dm.get_dataset_stats()
        
        assert stats['train_size'] == 2
        assert stats['val_size'] == 2
        assert stats['test_size'] == 2