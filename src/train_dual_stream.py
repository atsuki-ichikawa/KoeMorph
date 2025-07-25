"""
Training script for dual-stream KoeMorph model.

Implements training pipeline for the dual-stream architecture with 
separate mel-spectrogram and emotion2vec processing.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import KoeMorphDataModule
from src.model.simplified_dual_stream_model import SimplifiedDualStreamModel
from src.model.losses import BlendshapeMetrics, KoeMorphLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualStreamTrainer:
    """
    Trainer class for dual-stream KoeMorph model.
    
    Handles training with separate mel and emotion streams,
    attention visualization, and stream-specific metrics.
    """

    def __init__(self, config: DictConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = self._setup_device()

        # Initialize model and components
        self.model = self._setup_model()
        self.loss_fn = self._setup_loss_function()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Data
        self.data_module = self._setup_data()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Logging
        self.writer = SummaryWriter(log_dir="logs/dual_stream")
        self.metrics = BlendshapeMetrics()

        # Checkpointing
        self.checkpoint_dir = Path("checkpoints/dual_stream")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dual-stream trainer initialized.")
        logger.info(f"Model info: {self.model.get_model_info()}")

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.get("device", "auto") == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        logger.info(f"Using device: {device}")
        return device

    def _setup_model(self) -> nn.Module:
        """Setup dual-stream KoeMorph model."""
        model_config = self.config.model
        
        # Initialize monitoring if enabled
        monitoring_config = self.config.get("monitoring", {})
        if monitoring_config.get("enable", True):
            from src.utils.emotion_monitor import initialize_monitor
            self.monitor = initialize_monitor(monitoring_config)
            logger.info("Emotion processing monitoring enabled")
        else:
            self.monitor = None
        
        model = SimplifiedDualStreamModel(
            d_model=model_config.get("d_model", 256),
            num_heads=model_config.get("num_heads", 8),
            num_blendshapes=model_config.get("num_blendshapes", 52),
            sample_rate=model_config.get("sample_rate", 16000),
            target_fps=model_config.get("target_fps", 30),
            mel_sequence_length=model_config.get("mel_sequence_length", 256),
            emotion_config=model_config.get("emotion_config"),
            device=str(self.device),
        )
        model = model.to(self.device)

        # Load checkpoint if specified
        if self.config.get("checkpoint_path"):
            self._load_checkpoint(self.config.checkpoint_path, model)

        return model

    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function for dual-stream training."""
        loss_config = self.config.training.get("loss", {})
        
        # Use standard KoeMorph loss with additional stream-specific terms
        loss_fn = DualStreamLoss(
            l1_weight=loss_config.get("l1_weight", 1.0),
            l2_weight=loss_config.get("l2_weight", 0.1),
            velocity_weight=loss_config.get("velocity_weight", 0.05),
            stream_separation_weight=loss_config.get("stream_separation_weight", 0.01),
        )
        
        return loss_fn.to(self.device)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_config = self.config.training.optimizer
        
        if optimizer_config.name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.lr,
                betas=optimizer_config.get("betas", [0.9, 0.999]),
                weight_decay=optimizer_config.get("weight_decay", 1e-4),
            )
        elif optimizer_config.name == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.lr,
                betas=optimizer_config.get("betas", [0.9, 0.999]),
                weight_decay=optimizer_config.get("weight_decay", 1e-2),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

        return optimizer

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.training.get("scheduler")
        if not scheduler_config:
            return None

        if scheduler_config.name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
        elif scheduler_config.name == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config.name}")

        return scheduler

    def _setup_data(self) -> KoeMorphDataModule:
        """Setup data module."""
        data_module = KoeMorphDataModule(self.config.data)
        data_module.setup()
        return data_module

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data containing audio and blendshape targets
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Get inputs
        audio = batch["audio"].to(self.device)  # (B, T)
        target_blendshapes = batch["blendshapes"].to(self.device)  # (B, 52)
        
        # Forward pass with attention visualization
        output = self.model(audio, return_attention=True)
        pred_blendshapes = output["blendshapes"]
        
        # Compute loss
        loss_dict = self.loss_fn(
            predictions=pred_blendshapes,
            targets=target_blendshapes,
            mel_attention=output.get("mel_attention_weights"),
            emotion_attention=output.get("emotion_attention_weights"),
        )
        
        # Backward pass
        total_loss = loss_dict["total_loss"]
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=self.config.training.get("grad_clip_norm", 1.0)
        )
        
        self.optimizer.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        return loss_dict

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step.
        
        Args:
            batch: Batch of validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get inputs
            audio = batch["audio"].to(self.device)
            target_blendshapes = batch["blendshapes"].to(self.device)
            
            # Forward pass
            output = self.model(audio, return_attention=False)
            pred_blendshapes = output["blendshapes"]
            
            # Compute loss
            loss_dict = self.loss_fn(
                predictions=pred_blendshapes,
                targets=target_blendshapes,
            )
            
            # Compute metrics
            metrics = self.metrics(pred_blendshapes, target_blendshapes)
            
            # Combine loss and metrics
            result = {**loss_dict, **metrics}
            result = {k: v.item() if torch.is_tensor(v) else v for k, v in result.items()}
            
        return result

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        train_loader = self.data_module.train_dataloader()
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss_dict = self.train_step(batch)
            epoch_losses.append(loss_dict)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log step-level metrics
            if self.global_step % self.config.training.get("log_every_n_steps", 100) == 0:
                self._log_step_metrics(loss_dict)
            
            self.global_step += 1
            
            # Fast dev run
            if self.config.get("fast_dev_run", False) and batch_idx >= 2:
                break
        
        # Aggregate epoch losses
        epoch_metrics = self._aggregate_metrics(epoch_losses)
        
        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        
        val_loader = self.data_module.val_dataloader()
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss_dict = self.validation_step(batch)
            val_losses.append(loss_dict)
            
            # Update progress bar
            progress_bar.set_postfix({
                "val_loss": f"{loss_dict['total_loss']:.4f}"
            })
            
            # Fast dev run
            if self.config.get("fast_dev_run", False) and batch_idx >= 2:
                break
        
        # Aggregate validation metrics
        val_metrics = self._aggregate_metrics(val_losses)
        
        return val_metrics

    def _aggregate_metrics(self, metrics_list: list) -> Dict[str, float]:
        """Aggregate metrics from list of dictionaries."""
        if not metrics_list:
            return {}
        
        # Get all keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Average each metric
        aggregated = {}
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = sum(values) / len(values)
        
        return aggregated

    def _log_step_metrics(self, metrics: Dict[str, float]):
        """Log step-level metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"train_step/{key}", value, self.global_step)
        
        # Log learning rate
        self.writer.add_scalar(
            "train_step/lr", 
            self.optimizer.param_groups[0]['lr'], 
            self.global_step
        )

    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch-level metrics to tensorboard."""
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train_epoch/{key}", value, self.current_epoch)
        
        # Log validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val_epoch/{key}", value, self.current_epoch)

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": OmegaConf.to_container(self.config),
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved to {best_path}")

    def _load_checkpoint(self, checkpoint_path: str, model: nn.Module):
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    def fit(self):
        """Complete training loop."""
        logger.info("Starting dual-stream training...")
        
        max_epochs = self.config.training.get("max_epochs", 100)
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Reset temporal state for new epoch
            self.model.reset_temporal_state()
            
            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} training - Loss: {train_metrics['total_loss']:.4f}")
            
            # Validation epoch
            val_metrics = self.validate_epoch()
            logger.info(f"Epoch {epoch} validation - Loss: {val_metrics['total_loss']:.4f}")
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics["total_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["total_loss"]
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            # TODO: Implement early stopping logic
            
        logger.info("Training completed!")
        self.writer.close()


class DualStreamLoss(nn.Module):
    """
    Loss function for dual-stream model with stream separation regularization.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.1,
        velocity_weight: float = 0.05,
        stream_separation_weight: float = 0.01,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.velocity_weight = velocity_weight
        self.stream_separation_weight = stream_separation_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mel_attention: Optional[torch.Tensor] = None,
        emotion_attention: Optional[torch.Tensor] = None,
        prev_predictions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual-stream loss.
        
        Args:
            predictions: Predicted blendshapes (B, 52)
            targets: Target blendshapes (B, 52)
            mel_attention: Attention weights from mel stream (optional)
            emotion_attention: Attention weights from emotion stream (optional)
            prev_predictions: Previous predictions for velocity loss (optional)
            
        Returns:
            Dictionary of loss components
        """
        # Basic losses
        l1_loss = self.l1_loss(predictions, targets)
        l2_loss = self.l2_loss(predictions, targets)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        loss_dict = {
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
        }
        
        # Velocity loss (temporal smoothness)
        if prev_predictions is not None and self.velocity_weight > 0:
            velocity_loss = self.l2_loss(
                predictions - prev_predictions,
                targets - prev_predictions
            )
            loss_dict["velocity_loss"] = velocity_loss
            loss_dict["total_loss"] += self.velocity_weight * velocity_loss
        
        # Stream separation loss (encourage specialization)
        if (mel_attention is not None and emotion_attention is not None and 
            self.stream_separation_weight > 0):
            
            # Encourage mel stream to focus on mouth blendshapes
            # and emotion stream to focus on expression blendshapes
            from .dual_stream_attention import MOUTH_INDICES, EXPRESSION_INDICES
            
            mouth_loss = predictions[:, MOUTH_INDICES]
            expression_loss = predictions[:, EXPRESSION_INDICES]
            
            # Simple separation regularization
            separation_loss = torch.mean(torch.abs(
                torch.mean(mouth_loss, dim=1) - torch.mean(expression_loss, dim=1)
            ))
            
            loss_dict["separation_loss"] = separation_loss
            loss_dict["total_loss"] += self.stream_separation_weight * separation_loss
        
        return loss_dict


@hydra.main(version_base=None, config_path="../configs", config_name="dual_stream_config")
def main(config: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting dual-stream KoeMorph training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
    
    # Set random seed
    torch.manual_seed(config.get("seed", 42))
    
    # Initialize trainer
    trainer = DualStreamTrainer(config)
    
    # Start training
    trainer.fit()


if __name__ == "__main__":
    main()