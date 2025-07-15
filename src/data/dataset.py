"""
PyTorch Dataset and DataModule for ARKit blendshape data.

Unit name     : KoeMorphDataset
Input         : Directory paths with paired jsonl/wav files
Output        : Batched tensors {'wav': (B,L), 'arkit': (B,T,52)}
Dependencies  : torch, .io module
Assumptions   : Paired files with same basename, synchronized data
Failure modes : Missing files, memory issues with long sequences
Test cases    : test_dataset_len, test_getitem, test_dataloader_batch
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .io import ARKitDataLoader


class KoeMorphDataset(Dataset):
    """PyTorch Dataset for ARKit blendshape and audio data."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        max_audio_length: Optional[float] = None,
        file_pattern: str = "*.jsonl",
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing paired jsonl and wav files
            sample_rate: Audio sample rate (Hz)
            target_fps: Blendshape frame rate (FPS)
            max_audio_length: Maximum audio length in seconds (None for no limit)
            file_pattern: Pattern to match blendshape files
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.max_audio_length = max_audio_length

        # Initialize data loader
        self.loader = ARKitDataLoader(sample_rate=sample_rate, target_fps=target_fps)

        # Find paired files
        self.file_pairs = self._find_paired_files(file_pattern)

        if not self.file_pairs:
            raise ValueError(f"No paired files found in {data_dir}")

    def _find_paired_files(self, pattern: str) -> List[Tuple[Path, Path]]:
        """Find paired jsonl and wav files."""
        pairs = []

        # Find all jsonl files
        jsonl_files = list(self.data_dir.glob(pattern))

        for jsonl_path in jsonl_files:
            # Look for corresponding wav file
            wav_path = jsonl_path.with_suffix(".wav")

            if wav_path.exists():
                pairs.append((jsonl_path, wav_path))
            else:
                warnings.warn(f"No corresponding wav file for {jsonl_path}")

        return pairs

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        jsonl_path, wav_path = self.file_pairs[idx]

        try:
            sample = self.loader.load_sample(jsonl_path, wav_path)
        except Exception as e:
            warnings.warn(f"Failed to load sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the batch
            return self._get_dummy_sample()

        # Apply length limiting if specified
        if self.max_audio_length is not None:
            sample = self._limit_length(sample)

        return sample

    def _limit_length(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Limit sample length to max_audio_length."""
        max_audio_samples = int(self.max_audio_length * self.sample_rate)
        max_blendshape_frames = int(self.max_audio_length * self.target_fps)

        # Truncate if too long
        if len(sample["wav"]) > max_audio_samples:
            sample["wav"] = sample["wav"][:max_audio_samples]

        if len(sample["arkit"]) > max_blendshape_frames:
            sample["arkit"] = sample["arkit"][:max_blendshape_frames]

        return sample

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a dummy sample for error recovery."""
        dummy_audio_length = int(0.1 * self.sample_rate)  # 0.1 seconds
        dummy_bs_length = int(0.1 * self.target_fps)  # 0.1 seconds

        return {
            "wav": torch.zeros(dummy_audio_length),
            "arkit": torch.zeros(dummy_bs_length, 52),
        }

    def get_sample_info(self, idx: int) -> Dict[str, Union[str, float]]:
        """Get information about a sample without loading it."""
        jsonl_path, wav_path = self.file_pairs[idx]

        return {
            "jsonl_path": str(jsonl_path),
            "wav_path": str(wav_path),
            "basename": jsonl_path.stem,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with padding
    """
    # Separate audio and blendshape tensors
    audio_tensors = [sample["wav"] for sample in batch]
    blendshape_tensors = [sample["arkit"] for sample in batch]

    # Pad sequences to max length in batch
    audio_padded = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
    blendshape_padded = pad_sequence(
        blendshape_tensors, batch_first=True, padding_value=0.0
    )

    # Create attention masks (True for real data, False for padding)
    audio_lengths = torch.tensor([len(seq) for seq in audio_tensors])
    bs_lengths = torch.tensor([len(seq) for seq in blendshape_tensors])

    batch_size = len(batch)
    max_audio_len = audio_padded.size(1)
    max_bs_len = blendshape_padded.size(1)

    # Create masks
    audio_mask = torch.zeros(batch_size, max_audio_len, dtype=torch.bool)
    bs_mask = torch.zeros(batch_size, max_bs_len, dtype=torch.bool)

    for i, (audio_len, bs_len) in enumerate(zip(audio_lengths, bs_lengths)):
        audio_mask[i, :audio_len] = True
        bs_mask[i, :bs_len] = True

    return {
        "wav": audio_padded,
        "arkit": blendshape_padded,
        "audio_mask": audio_mask,
        "blendshape_mask": bs_mask,
        "audio_lengths": audio_lengths,
        "blendshape_lengths": bs_lengths,
    }


class KoeMorphDataModule:
    """
    Data module that handles train/val/test splits and DataLoaders.

    This class follows PyTorch Lightning DataModule pattern but works
    with standard PyTorch training loops.
    """

    def __init__(
        self,
        train_data_dir: Union[str, Path],
        val_data_dir: Optional[Union[str, Path]] = None,
        test_data_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        sample_rate: int = 16000,
        target_fps: float = 30.0,
        max_audio_length: Optional[float] = None,
    ):
        """
        Initialize the data module.

        Args:
            train_data_dir: Training data directory
            val_data_dir: Validation data directory (optional)
            test_data_dir: Test data directory (optional)
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            sample_rate: Audio sample rate
            target_fps: Blendshape frame rate
            max_audio_length: Maximum audio length in seconds
        """
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.max_audio_length = max_audio_length

        # Datasets will be created lazily
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self):
        """Setup datasets."""
        # Create training dataset
        self._train_dataset = KoeMorphDataset(
            data_dir=self.train_data_dir,
            sample_rate=self.sample_rate,
            target_fps=self.target_fps,
            max_audio_length=self.max_audio_length,
        )

        # Create validation dataset if specified
        if self.val_data_dir is not None:
            self._val_dataset = KoeMorphDataset(
                data_dir=self.val_data_dir,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                max_audio_length=self.max_audio_length,
            )

        # Create test dataset if specified
        if self.test_data_dir is not None:
            self._test_dataset = KoeMorphDataset(
                data_dir=self.test_data_dir,
                sample_rate=self.sample_rate,
                target_fps=self.target_fps,
                max_audio_length=self.max_audio_length,
            )

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self._train_dataset is None:
            self.setup()

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,  # Avoid issues with batch normalization
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation DataLoader."""
        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test DataLoader."""
        if self._test_dataset is None:
            return None

        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about the datasets."""
        if self._train_dataset is None:
            self.setup()

        stats = {
            "train_size": len(self._train_dataset) if self._train_dataset else 0,
            "val_size": len(self._val_dataset) if self._val_dataset else 0,
            "test_size": len(self._test_dataset) if self._test_dataset else 0,
        }

        return stats
