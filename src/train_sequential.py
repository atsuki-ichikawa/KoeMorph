"""
Sequential training script for dual-stream KoeMorph model.

This script implements time-series aware training that maintains
temporal continuity for proper temporal smoothing and emotion
feature consistency.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from src.model.sequential_dual_stream_model_fixed import SequentialDualStreamModelFixed as SequentialDualStreamModel
from src.data.sequential_dataset import create_sequential_dataloader
from src.data.koemorph_dataset import SequentialKoeMorphDataset
from src.model.losses import (
    KoeMorphLoss,
    PerceptualBlendshapeLoss,
    LandmarkConsistencyLoss
)

logger = logging.getLogger(__name__)


class SequentialTrainer:
    """
    Trainer for sequential time-series aware training.
    
    Maintains model state across batches from the same sequence
    to ensure proper temporal continuity.
    """
    
    def __init__(
        self,
        model: SequentialDualStreamModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize sequential trainer.
        
        Args:
            model: Dual-stream model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
            log_dir: TensorBoard log directory
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip = gradient_clip
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6,
        )
        
        # Loss functions (temporary simple MSE for debugging)
        self.criterion = nn.MSELoss()
        
        # TensorBoard
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Sequence tracking
        self.current_file_idx = None
        self.sequence_losses = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch maintaining temporal continuity.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'mse': 0.0,
            'smoothing': 0.0,
            'lip_sync': 0.0,
        }
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch['audio'].to(self.device)
            target_blendshapes = batch['blendshapes'].to(self.device)
            
            # Handle different batch formats
            if 'file_indices' in batch:
                file_indices = batch['file_indices']
            else:
                # New KoeMorph dataset format
                # Add batch dimension if needed
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                if target_blendshapes.dim() == 2:
                    target_blendshapes = target_blendshapes.unsqueeze(0)
                # Create dummy file indices
                file_indices = torch.zeros(audio.shape[0], dtype=torch.long)
            
            # Check if we need to reset temporal state
            if self.current_file_idx is None:
                self.current_file_idx = file_indices[0].item()
                self.model.reset_temporal_state()
            
            # Reset state when switching files
            for i, file_idx in enumerate(file_indices):
                if file_idx.item() != self.current_file_idx:
                    # Process accumulated sequence losses
                    if self.sequence_losses:
                        self._log_sequence_metrics()
                    
                    # Reset for new sequence
                    self.current_file_idx = file_idx.item()
                    self.model.reset_temporal_state()
                    self.sequence_losses = []
                    
                    # Note: This breaks temporal continuity within the batch
                    # In practice, the batch sampler should prevent this
                    logger.debug(f"File transition in batch at index {i}")
            
            # Forward pass
            outputs = self.model(audio, return_attention=False)
            pred_blendshapes = outputs['blendshapes']
            
            # Compute losses (simple MSE for debugging)
            print(f"DEBUG: pred_blendshapes shape: {pred_blendshapes.shape}")
            print(f"DEBUG: target_blendshapes shape: {target_blendshapes.shape}")
            
            total_loss = self.criterion(pred_blendshapes, target_blendshapes)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track losses (simplified for debugging)
            epoch_losses['total'] += total_loss.item()
            num_batches += 1
            
            # Track sequence losses
            self.sequence_losses.append(total_loss.item())
            
            # Update progress bar (simplified for debugging)
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
            })
            
            # Log to TensorBoard (simplified for debugging)
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', total_loss.item(), self.global_step)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Step scheduler
        self.scheduler.step()
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model maintaining temporal continuity.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'mse': 0.0,
            'smoothing': 0.0,
            'lip_sync': 0.0,
        }
        num_batches = 0
        
        # Track per-sequence metrics
        sequence_metrics = {}
        current_file_idx = None
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                audio = batch['audio'].to(self.device)
                target_blendshapes = batch['blendshapes'].to(self.device)
                
                # Handle different batch formats
                if 'file_indices' in batch:
                    file_indices = batch['file_indices']
                    file_names = batch['file_names']
                else:
                    # New KoeMorph dataset format
                    # Add batch dimension if needed
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    if target_blendshapes.dim() == 2:
                        target_blendshapes = target_blendshapes.unsqueeze(0)
                    # Create dummy file indices and names
                    file_indices = torch.zeros(audio.shape[0], dtype=torch.long)
                    file_names = [batch.get('recording_id', f'recording_{batch_idx}')] * audio.shape[0]
                
                # Reset state when switching files
                for i, file_idx in enumerate(file_indices):
                    if current_file_idx is None or file_idx.item() != current_file_idx:
                        current_file_idx = file_idx.item()
                        self.model.reset_temporal_state()
                        
                        # Initialize sequence metrics
                        if file_names[i] not in sequence_metrics:
                            sequence_metrics[file_names[i]] = {
                                'losses': [],
                                'smoothness': [],
                            }
                
                # Forward pass
                outputs = self.model(audio, return_attention=True)
                pred_blendshapes = outputs['blendshapes']
                
                # Compute losses
                loss_dict = self.criterion(
                    pred_blendshapes,
                    target_blendshapes,
                    return_components=True
                )
                
                # Track losses
                for key in val_losses:
                    val_losses[key] += loss_dict[key].item()
                num_batches += 1
                
                # Track per-sequence metrics
                for i, file_name in enumerate(file_names):
                    sequence_metrics[file_name]['losses'].append(
                        loss_dict['total'].item()
                    )
                    
                    # Calculate frame-to-frame smoothness
                    if pred_blendshapes.shape[1] > 1:
                        diff = torch.diff(pred_blendshapes[i], dim=0)
                        smoothness = torch.mean(torch.abs(diff)).item()
                        sequence_metrics[file_name]['smoothness'].append(smoothness)
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Calculate per-sequence statistics
        val_losses['sequence_stats'] = self._calculate_sequence_stats(sequence_metrics)
        
        # Reset model state after validation
        self.model.reset_temporal_state()
        
        return val_losses
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model.get_model_info(),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = path.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def _log_training_step(self, loss_dict: Dict, outputs: Dict, batch: Dict):
        """Log training metrics to TensorBoard."""
        if not self.writer:
            return
        
        # Log losses
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'train/loss_{key}', value.item(), self.global_step)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
        
        # Log gradient norms
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar('train/gradient_norm', total_norm, self.global_step)
        
        # Log attention weights periodically
        if self.global_step % 100 == 0 and 'mel_attention_weights' in outputs:
            # Log attention visualizations
            mel_attn = outputs['mel_attention_weights'][0]  # First sample
            emotion_attn = outputs['emotion_attention_weights'][0]
            
            self.writer.add_image(
                'attention/mel',
                mel_attn.cpu().numpy(),
                self.global_step,
                dataformats='HW'
            )
            self.writer.add_image(
                'attention/emotion',
                emotion_attn.cpu().numpy(),
                self.global_step,
                dataformats='HW'
            )
    
    def _log_sequence_metrics(self):
        """Log metrics for completed sequence."""
        if not self.sequence_losses:
            return
        
        # Calculate sequence statistics
        seq_mean_loss = np.mean(self.sequence_losses)
        seq_std_loss = np.std(self.sequence_losses)
        seq_trend = np.polyfit(range(len(self.sequence_losses)), self.sequence_losses, 1)[0]
        
        if self.writer:
            self.writer.add_scalar('sequence/mean_loss', seq_mean_loss, self.global_step)
            self.writer.add_scalar('sequence/std_loss', seq_std_loss, self.global_step)
            self.writer.add_scalar('sequence/loss_trend', seq_trend, self.global_step)
    
    def _calculate_sequence_stats(self, sequence_metrics: Dict) -> Dict:
        """Calculate statistics across sequences."""
        stats = {
            'mean_loss_per_sequence': {},
            'mean_smoothness_per_sequence': {},
            'overall_smoothness': 0.0,
        }
        
        all_smoothness = []
        
        for file_name, metrics in sequence_metrics.items():
            if metrics['losses']:
                stats['mean_loss_per_sequence'][file_name] = np.mean(metrics['losses'])
            
            if metrics['smoothness']:
                mean_smooth = np.mean(metrics['smoothness'])
                stats['mean_smoothness_per_sequence'][file_name] = mean_smooth
                all_smoothness.extend(metrics['smoothness'])
        
        if all_smoothness:
            stats['overall_smoothness'] = np.mean(all_smoothness)
        
        return stats


def create_dataloader(cfg: DictConfig, split: str = "train") -> DataLoader:
    """
    Create dataloader based on configuration type.
    
    Supports both old sequential dataset and new KoeMorph v2 dataset.
    """
    # Check if using new KoeMorph dataset
    if hasattr(cfg.data, '_target_') and 'koemorph' in cfg.data._target_.lower():
        logger.info(f"Using new KoeMorph v2 dataset for {split}")
        
        # Determine data directory
        if split == "train":
            data_dir = cfg.data.train_data_dir
        elif split == "val":
            data_dir = cfg.data.val_data_dir
        else:
            data_dir = cfg.data.test_data_dir
            
        # Create SequentialKoeMorphDataset
        sequential_config = cfg.data.get('sequential', {})
        window_frames = sequential_config.get('window_frames', 256)
        stride_frames = sequential_config.get('stride_frames', 128 if split == "train" else 256)
        
        dataset = SequentialKoeMorphDataset(
            data_dir=data_dir,
            window_frames=window_frames,
            stride_frames=stride_frames,
            sample_rate=cfg.data.sample_rate,
            target_fps=cfg.data.target_fps,
            shuffle_files=(split == "train"),
            loop_dataset=(split == "train"),
        )
        
        # Create DataLoader for IterableDataset
        return DataLoader(
            dataset,
            batch_size=1,  # IterableDataset handles batching
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
    else:
        # Use old sequential dataloader
        logger.info(f"Using old sequential dataset for {split}")
        
        if split == "train":
            return create_sequential_dataloader(
                data_dir=cfg.data.train_data_dir,
                batch_size=cfg.data.batch_size,
                window_frames=cfg.data.get('window_frames', 256),
                stride_frames=cfg.data.get('stride_frames', 128),
                num_workers=cfg.data.num_workers,
                shuffle_files=True,
                loop_dataset=True,
            )
        else:
            return create_sequential_dataloader(
                data_dir=cfg.data.val_data_dir,
                batch_size=cfg.data.batch_size,
                window_frames=cfg.data.get('window_frames', 256),
                stride_frames=cfg.data.get('stride_frames', 256),
                num_workers=cfg.data.num_workers,
                shuffle_files=False,
                loop_dataset=False,
            )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_loader = create_dataloader(cfg, split="train")
    
    val_loader = None
    if cfg.data.val_data_dir:
        val_loader = create_dataloader(cfg, split="val")
    
    # Create model
    logger.info("Creating dual-stream model...")
    
    # Update model config for sequential training
    from omegaconf import OmegaConf
    model_config = OmegaConf.create(cfg.model)
    OmegaConf.set_struct(model_config, False)  # Allow dynamic keys
    model_config['real_time_mode'] = False  # Use batch mode for training
    
    # Determine target FPS from data configuration
    target_fps = cfg.data.get('target_fps', cfg.get('frame_rate', 30))
    logger.info(f"Using target FPS: {target_fps}")
    
    # Get window frames from data configuration
    sequential_config = cfg.data.get('sequential', {})
    window_frames = sequential_config.get('window_frames', 256)
    
    model = SequentialDualStreamModel(
        d_model=model_config.get('d_model', 256),
        num_heads=model_config.attention.get('num_heads', 8),
        num_blendshapes=52,  # ARKit blendshapes count
        sample_rate=cfg.data.sample_rate,
        target_fps=target_fps,
        mel_sequence_length=window_frames,
        emotion_config=model_config.get('emotion_config', {}),
        device=str(device),
        real_time_mode=model_config.get('real_time_mode', False),
        stride_frames=1,  # Dense output for all frames in sequence
    )
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model initialized: {model_info['total_parameters']:,} parameters")
    logger.info(f"Emotion backend: {model_info['emotion_backend']}")
    
    # Create trainer
    trainer = SequentialTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=cfg.training.optimizer.get('lr', 1e-4),
        weight_decay=cfg.training.optimizer.get('weight_decay', 1e-5),
        gradient_clip=cfg.training.get('gradient_clip_val', 1.0),
        log_dir=Path("logs") if not cfg.training.get('log_dir') else Path(cfg.training.log_dir),
    )
    
    # Load checkpoint if resuming
    if cfg.training.get('resume_from'):
        trainer.load_checkpoint(Path(cfg.training.resume_from))
    
    # Training loop
    logger.info("Starting sequential training...")
    
    for epoch in range(trainer.epoch, cfg.training.max_epochs):
        trainer.epoch = epoch
        
        # Train
        train_start = time.time()
        train_metrics = trainer.train_epoch()
        train_time = time.time() - train_start
        
        logger.info(
            f"Epoch {epoch} - Train: "
            f"loss={train_metrics['total']:.4f}, "
            f"mse={train_metrics['mse']:.4f}, "
            f"smooth={train_metrics['smoothing']:.4f}, "
            f"time={train_time:.1f}s"
        )
        
        # Validate
        if val_loader and epoch % cfg.training.get('val_frequency', 1) == 0:
            val_start = time.time()
            val_metrics = trainer.validate()
            val_time = time.time() - val_start
            
            logger.info(
                f"Epoch {epoch} - Val: "
                f"loss={val_metrics['total']:.4f}, "
                f"smoothness={val_metrics['sequence_stats']['overall_smoothness']:.4f}, "
                f"time={val_time:.1f}s"
            )
            
            # Save best model
            if val_metrics['total'] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics['total']
                trainer.save_checkpoint(
                    Path(cfg.training.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth',
                    is_best=True
                )
        
        # Save periodic checkpoints
        if epoch % cfg.training.get('checkpoint_frequency', 10) == 0:
            trainer.save_checkpoint(
                Path(cfg.training.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
            )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()