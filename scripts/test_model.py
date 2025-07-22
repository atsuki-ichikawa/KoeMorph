#!/usr/bin/env python3
"""Test trained KoeMorph model on test dataset."""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append("/home/ichikawa/KoeMorph")

from src.data.dataset import KoeMorphDataModule
from src.model.simplified_model import SimplifiedKoeMorphModel
from src.model.losses import BlendshapeMetrics, KoeMorphLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(
    checkpoint_path: str,
    config_path: str = None,
    tensorboard_dir: str = None,
    save_predictions: bool = False,
) -> Dict[str, float]:
    """Test model on test dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file (uses checkpoint config if None)
        tensorboard_dir: Optional directory for TensorBoard logging
        save_predictions: Whether to save predictions to file
        
    Returns:
        Dictionary of test metrics
    """
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Load config
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create(checkpoint.get("config", {}))
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model with proper config
    if hasattr(config, 'model'):
        model_config = dict(config.model)
        # Remove _target_ key if present
        model_config.pop('_target_', None)
    else:
        model_config = {}
    
    model = SimplifiedKoeMorphModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Initialize data module with correct parameters
    data_config = OmegaConf.to_container(config.data, resolve=True)
    
    # Extract only the parameters that KoeMorphDataModule expects
    data_module = KoeMorphDataModule(
        train_data_dir=data_config.get('train_data_dir'),
        val_data_dir=data_config.get('val_data_dir'),
        test_data_dir=data_config.get('test_data_dir'),
        batch_size=data_config.get('batch_size', 16),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        sample_rate=data_config.get('sample_rate', 16000),
        target_fps=data_config.get('target_fps', 30),
        max_audio_length=data_config.get('audio_max_length', 10.0)
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    if test_loader is None:
        logger.error("No test dataset found!")
        return {}
    
    # Initialize metrics and loss
    metrics = BlendshapeMetrics()
    
    # Get loss config from training config or use defaults
    if hasattr(config, 'training') and hasattr(config.training, 'loss'):
        loss_config = dict(config.training.loss)
        loss_config.pop('_target_', None)
    else:
        # Use default loss config
        loss_config = {
            'mse_weight': 1.0,
            'l1_weight': 0.1,
            'perceptual_weight': 0.5
        }
    
    loss_fn = KoeMorphLoss(**loss_config)
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(device)
    test_losses = []
    
    # Initialize TensorBoard if requested
    writer = None
    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard logging to {tensorboard_dir}")
    
    # Storage for predictions if requested
    all_predictions = []
    all_targets = []
    
    # Test loop
    logger.info("Starting test on test dataset...")
    logger.info(f"Test dataset path: {config.data.test_data_dir}")
    logger.info(f"Number of test batches: {len(test_loader)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            audio = batch["wav"].to(device)
            target_blendshapes = batch["arkit"].to(device)
            
            # Handle 3D tensor case
            if target_blendshapes.dim() == 3:
                target_blendshapes = target_blendshapes[:, 0, :]
            
            # Forward pass
            pred_blendshapes = model(audio)
            
            # Compute loss
            loss = loss_fn(pred_blendshapes, target_blendshapes)
            if isinstance(loss, tuple):
                loss = loss[0]
            test_losses.append(loss.item())
            
            # Update metrics
            metrics.update(pred_blendshapes, target_blendshapes)
            
            # Store predictions if requested
            if save_predictions:
                all_predictions.append(pred_blendshapes.cpu())
                all_targets.append(target_blendshapes.cpu())
            
            # Log batch metrics to TensorBoard
            if writer and batch_idx % 10 == 0:
                writer.add_scalar("test/batch_loss", loss.item(), batch_idx)
    
    # Compute final metrics
    final_metrics = metrics.compute()
    final_metrics["loss"] = sum(test_losses) / len(test_losses) if test_losses else 0.0
    
    # Log final metrics
    logger.info("\nTest Results on Test Dataset:")
    logger.info("-" * 50)
    for metric_name, value in final_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        if writer:
            writer.add_scalar(f"test/final_{metric_name}", value, 0)
    
    # Save predictions if requested
    if save_predictions:
        predictions_path = Path(checkpoint_path).parent / "test_predictions.pt"
        torch.save({
            "predictions": torch.cat(all_predictions),
            "targets": torch.cat(all_targets),
            "metrics": final_metrics
        }, predictions_path)
        logger.info(f"Saved predictions to {predictions_path}")
    
    # Add histogram of predictions to TensorBoard
    if writer and all_predictions:
        all_preds = torch.cat(all_predictions)
        all_tgts = torch.cat(all_targets)
        
        # Add histograms for each blendshape
        num_blendshapes = model_config.get('num_blendshapes', 52)
        for i in range(num_blendshapes):
            writer.add_histogram(f"test/blendshape_{i}_predictions", all_preds[:, i], 0)
            writer.add_histogram(f"test/blendshape_{i}_targets", all_tgts[:, i], 0)
    
    if writer:
        writer.close()
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Test KoeMorph model on test data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (uses checkpoint config if not provided)"
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        help="Directory for TensorBoard logging"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file"
    )
    
    args = parser.parse_args()
    
    # Run test
    test_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tensorboard_dir=args.tensorboard_dir,
        save_predictions=args.save_predictions
    )


if __name__ == "__main__":
    main()