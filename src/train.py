"""
Training script for KoeMorph model.

Implements complete training pipeline with Hydra configuration,
logging, checkpointing, and evaluation.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import KoeMorphDataModule
from src.features.emotion2vec import Emotion2VecExtractor
from src.features.prosody import ProsodyExtractor
from src.features.stft import MelSpectrogramExtractor
from src.model.gaussian_face import create_koemorph_model
from src.model.losses import BlendshapeMetrics, KoeMorphLoss


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KoeMorphTrainer:
    """
    Trainer class for KoeMorph model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = self._setup_device()
        
        # Initialize model and components
        self.model = self._setup_model()
        self.feature_extractors = self._setup_feature_extractors()
        self.loss_fn = self._setup_loss_function()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Data
        self.data_module = self._setup_data()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = SummaryWriter(log_dir='logs')
        self.metrics = BlendshapeMetrics()
        
        # Checkpointing
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Trainer initialized. Model has {self.model.get_num_parameters():,} parameters")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.get('device', 'auto') == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_model(self) -> nn.Module:
        """Setup KoeMorph model."""
        model = create_koemorph_model(self.config.model)
        model = model.to(self.device)
        
        # Load checkpoint if specified
        if self.config.get('checkpoint_path'):
            self._load_checkpoint(self.config.checkpoint_path)
        
        return model
    
    def _setup_feature_extractors(self) -> Dict[str, nn.Module]:
        """Setup feature extraction modules."""
        extractors = {}
        
        # Mel-spectrogram extractor
        extractors['mel'] = MelSpectrogramExtractor(
            sample_rate=self.config.data.sample_rate,
            target_fps=self.config.data.target_fps,
            n_fft=self.config.model.mel.n_fft,
            n_mels=self.config.model.mel.n_mels,
            f_min=self.config.model.mel.f_min,
            f_max=self.config.model.mel.f_max,
        ).to(self.device)
        
        # Prosody extractor
        extractors['prosody'] = ProsodyExtractor(
            sample_rate=self.config.data.sample_rate,
            target_fps=self.config.data.target_fps,
            frame_length=self.config.model.prosody.frame_length,
            frame_shift=self.config.model.prosody.frame_shift,
        ).to(self.device)
        
        # Emotion2vec extractor
        extractors['emotion2vec'] = Emotion2VecExtractor(
            model_name=self.config.model.emotion2vec.model_name,
            target_fps=self.config.data.target_fps,
            sample_rate=self.config.data.sample_rate,
            freeze_pretrained=self.config.model.emotion2vec.freeze_pretrained,
            output_dim=self.config.model.emotion_dim,
        ).to(self.device)
        
        return extractors
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        return KoeMorphLoss(
            mse_weight=self.config.training.loss.mse_weight,
            l1_weight=self.config.training.loss.l1_weight,
            perceptual_weight=self.config.training.loss.perceptual_weight,
            temporal_weight=self.config.training.loss.get('temporal_weight', 0.2),
            sparsity_weight=self.config.training.loss.get('sparsity_weight', 0.01),
            smoothness_weight=self.config.training.loss.get('smoothness_weight', 0.1),
        )
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_config = self.config.training.optimizer
        
        if optimizer_config._target_ == 'torch.optim.AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config._target_}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if not hasattr(self.config.training, 'lr_scheduler'):
            return None
        
        scheduler_config = self.config.training.lr_scheduler
        
        if scheduler_config._target_ == 'torch.optim.lr_scheduler.CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config._target_}")
    
    def _setup_data(self) -> KoeMorphDataModule:
        """Setup data module."""
        data_module = KoeMorphDataModule(
            train_data_dir=self.config.data.train_data_dir,
            val_data_dir=self.config.data.get('val_data_dir'),
            test_data_dir=self.config.data.get('test_data_dir'),
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            sample_rate=self.config.data.sample_rate,
            target_fps=self.config.data.target_fps,
            max_audio_length=self.config.data.get('audio_max_length'),
        )
        
        data_module.setup()
        return data_module
    
    def _extract_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract multi-stream features from audio."""
        audio = batch['wav'].to(self.device)  # (B, L)
        
        # Extract features
        with torch.no_grad():
            mel_features = self.feature_extractors['mel'](audio)
            prosody_features = self.feature_extractors['prosody'](audio)
            emotion_features = self.feature_extractors['emotion2vec'](audio)
        
        return {
            'mel': mel_features,
            'prosody': prosody_features,
            'emotion': emotion_features,
            'audio_mask': batch.get('audio_mask'),
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        epoch_losses = []
        epoch_metrics = {}
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            target_blendshapes = batch['arkit'].to(self.device)  # (B, T, 52)
            
            # For simplicity, use first frame as target (could be improved)
            if target_blendshapes.dim() == 3:
                target_blendshapes = target_blendshapes[:, 0, :]  # (B, 52)
            
            # Extract features
            features = self._extract_features(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                mel_features=features['mel'],
                prosody_features=features['prosody'],
                emotion_features=features['emotion'],
                audio_mask=features['audio_mask'],
                prev_blendshapes=None,  # Could use previous predictions
            )
            
            pred_blendshapes = output['blendshapes']
            
            # Compute loss
            loss, loss_metrics = self.loss_fn(pred_blendshapes, target_blendshapes)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.get('gradient_clip_val'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.metrics.update(pred_blendshapes, target_blendshapes)
            epoch_losses.append(loss.item())
            
            # Accumulate loss metrics
            for key, value in loss_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if self.global_step % self.config.training.log_every_n_steps == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for key, value in loss_metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            self.global_step += 1
            
            # Early break for debugging
            if self.config.get('debug') and batch_idx > 5:
                break
        
        # Compute epoch metrics
        computed_metrics = self.metrics.compute()
        epoch_metrics.update(computed_metrics)
        
        # Average losses
        for key in epoch_metrics:
            if isinstance(epoch_metrics[key], list):
                epoch_metrics[key] = sum(epoch_metrics[key]) / len(epoch_metrics[key])
        
        epoch_metrics['loss'] = sum(epoch_losses) / len(epoch_losses)
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if not hasattr(self.data_module, '_val_dataset') or self.data_module._val_dataset is None:
            return {}
        
        self.model.eval()
        val_metrics = BlendshapeMetrics()
        val_losses = []
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                # Move data to device
                target_blendshapes = batch['arkit'].to(self.device)
                
                if target_blendshapes.dim() == 3:
                    target_blendshapes = target_blendshapes[:, 0, :]
                
                # Extract features
                features = self._extract_features(batch)
                
                # Forward pass
                output = self.model(
                    mel_features=features['mel'],
                    prosody_features=features['prosody'],
                    emotion_features=features['emotion'],
                    audio_mask=features['audio_mask'],
                    prev_blendshapes=None,
                )
                
                pred_blendshapes = output['blendshapes']
                
                # Compute loss
                loss, _ = self.loss_fn(pred_blendshapes, target_blendshapes)
                val_losses.append(loss.item())
                
                # Update metrics
                val_metrics.update(pred_blendshapes, target_blendshapes)
                
                # Early break for debugging
                if self.config.get('debug') and batch_idx > 2:
                    break
        
        # Compute validation metrics
        computed_metrics = val_metrics.compute()
        computed_metrics['loss'] = sum(val_losses) / len(val_losses) if val_losses else 0.0
        
        return computed_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': OmegaConf.to_container(self.config),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {metrics.get('loss', 0):.4f}")
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # Reset temporal state
            self.model.reset_temporal_state()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics.get('mae', 0):.4f}")
            
            if val_metrics:
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics.get('mae', 0):.4f}")
                
                # Tensorboard logging
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train_epoch/{key}', value, epoch)
            
            # Checkpointing
            current_val_loss = val_metrics.get('loss', float('inf'))
            is_best = current_val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = current_val_loss
            
            # Save checkpoint every N epochs or if best
            if (epoch + 1) % self.config.training.get('save_every_n_epochs', 5) == 0 or is_best:
                self.save_checkpoint(val_metrics or train_metrics, is_best=is_best)
            
            # Early stopping
            if hasattr(self.config.training, 'early_stopping'):
                # Implement early stopping logic here if needed
                pass
        
        logger.info("Training completed!")
        self.writer.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """Main training function."""
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Set random seeds for reproducibility
    if config.get('seed'):
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    # Create trainer and start training
    trainer = KoeMorphTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()