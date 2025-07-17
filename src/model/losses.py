"""
Loss functions and metrics for blendshape generation.

Unit name     : KoeMorphLoss
Input         : pred_blendshapes (B,52), target_blendshapes (B,52)
Output        : scalar loss + metrics dict
Dependencies  : torch.nn, librosa (for perceptual metrics)
Assumptions   : Blendshapes in [0,1] range, synchronized sequences
Failure modes : Gradient explosion, metric computation errors
Test cases    : test_loss_computation, test_metric_calculation, test_gradient_flow
"""

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not available. Some perceptual metrics will be unavailable.")


class KoeMorphLoss(nn.Module):
    """
    Combined loss function for blendshape generation.

    Combines reconstruction loss, perceptual loss, temporal consistency,
    and regularization terms for high-quality blendshape prediction.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.1,
        perceptual_weight: float = 0.5,
        temporal_weight: float = 0.2,
        sparsity_weight: float = 0.01,
        smoothness_weight: float = 0.1,
        landmark_weight: float = 0.3,
        velocity_weight: float = 0.05,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Initialize combined loss function.

        Args:
            mse_weight: Weight for MSE reconstruction loss
            l1_weight: Weight for L1 reconstruction loss
            perceptual_weight: Weight for perceptual consistency loss
            temporal_weight: Weight for temporal consistency loss
            sparsity_weight: Weight for sparsity regularization
            smoothness_weight: Weight for smoothness regularization
            landmark_weight: Weight for landmark-based loss
            velocity_weight: Weight for velocity consistency
            reduction: Reduction method ('mean', 'sum', 'none')
            eps: Small value for numerical stability
        """
        super().__init__()

        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
        self.landmark_weight = landmark_weight
        self.velocity_weight = velocity_weight
        self.reduction = reduction
        self.eps = eps

        # Loss components
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.l1_loss = nn.L1Loss(reduction=reduction)

        # Perceptual loss components
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualBlendshapeLoss()

        # Landmark loss
        if landmark_weight > 0:
            self.landmark_loss = LandmarkConsistencyLoss()

    def forward(
        self,
        pred_blendshapes: torch.Tensor,
        target_blendshapes: torch.Tensor,
        prev_pred: Optional[torch.Tensor] = None,
        prev_target: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss and metrics.

        Args:
            pred_blendshapes: Predicted blendshapes of shape (B, 52)
            target_blendshapes: Target blendshapes of shape (B, 52)
            prev_pred: Previous predicted blendshapes for temporal loss
            prev_target: Previous target blendshapes for temporal loss
            audio_features: Audio features for perceptual loss

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        losses = {}
        metrics = {}

        # Basic reconstruction losses
        if self.mse_weight > 0:
            mse_loss = self.mse_loss(pred_blendshapes, target_blendshapes)
            losses["mse"] = self.mse_weight * mse_loss
            metrics["mse"] = mse_loss.item()

        if self.l1_weight > 0:
            l1_loss = self.l1_loss(pred_blendshapes, target_blendshapes)
            losses["l1"] = self.l1_weight * l1_loss
            metrics["l1"] = l1_loss.item()

        # Perceptual loss
        if self.perceptual_weight > 0 and hasattr(self, "perceptual_loss"):
            perceptual_loss = self.perceptual_loss(
                pred_blendshapes, target_blendshapes, audio_features
            )
            losses["perceptual"] = self.perceptual_weight * perceptual_loss
            metrics["perceptual"] = perceptual_loss.item()

        # Temporal consistency loss
        if (
            self.temporal_weight > 0
            and prev_pred is not None
            and prev_target is not None
        ):
            temporal_loss = self._compute_temporal_loss(
                pred_blendshapes, target_blendshapes, prev_pred, prev_target
            )
            losses["temporal"] = self.temporal_weight * temporal_loss
            metrics["temporal"] = temporal_loss.item()

        # Velocity consistency loss
        if (
            self.velocity_weight > 0
            and prev_pred is not None
            and prev_target is not None
        ):
            velocity_loss = self._compute_velocity_loss(
                pred_blendshapes, target_blendshapes, prev_pred, prev_target
            )
            losses["velocity"] = self.velocity_weight * velocity_loss
            metrics["velocity"] = velocity_loss.item()

        # Sparsity regularization
        if self.sparsity_weight > 0:
            sparsity_loss = self._compute_sparsity_loss(pred_blendshapes)
            losses["sparsity"] = self.sparsity_weight * sparsity_loss
            metrics["sparsity"] = sparsity_loss.item()

        # Smoothness regularization
        if self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss(pred_blendshapes)
            losses["smoothness"] = self.smoothness_weight * smoothness_loss
            metrics["smoothness"] = smoothness_loss.item()

        # Landmark consistency loss
        if self.landmark_weight > 0 and hasattr(self, "landmark_loss"):
            landmark_loss = self.landmark_loss(pred_blendshapes, target_blendshapes)
            losses["landmark"] = self.landmark_weight * landmark_loss
            metrics["landmark"] = landmark_loss.item()

        # Combine all losses
        total_loss = sum(losses.values())

        # Additional metrics
        metrics.update(
            self._compute_additional_metrics(pred_blendshapes, target_blendshapes)
        )

        return total_loss, metrics

    def _compute_temporal_loss(
        self,
        pred_current: torch.Tensor,
        target_current: torch.Tensor,
        pred_prev: torch.Tensor,
        target_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Compute differences
        pred_diff = pred_current - pred_prev
        target_diff = target_current - target_prev

        # L2 loss on temporal differences
        temporal_loss = F.mse_loss(pred_diff, target_diff)

        return temporal_loss

    def _compute_velocity_loss(
        self,
        pred_current: torch.Tensor,
        target_current: torch.Tensor,
        pred_prev: torch.Tensor,
        target_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity consistency loss."""
        # Compute velocities
        pred_velocity = pred_current - pred_prev
        target_velocity = target_current - target_prev

        # L1 loss on velocities (more robust to outliers)
        velocity_loss = F.l1_loss(pred_velocity, target_velocity)

        return velocity_loss

    def _compute_sparsity_loss(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        # L1 norm to encourage sparsity
        sparsity_loss = torch.mean(torch.abs(blendshapes))

        return sparsity_loss

    def _compute_smoothness_loss(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """Compute smoothness regularization loss."""
        # Total variation loss - encourages smoothness across blendshapes
        diff = torch.diff(
            blendshapes, dim=1
        )  # Differences between adjacent blendshapes
        smoothness_loss = torch.mean(torch.abs(diff))

        return smoothness_loss

    def _compute_additional_metrics(
        self,
        pred_blendshapes: torch.Tensor,
        target_blendshapes: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute additional evaluation metrics."""
        metrics = {}

        with torch.no_grad():
            # Mean Absolute Error
            mae = F.l1_loss(pred_blendshapes, target_blendshapes)
            metrics["mae"] = mae.item()

            # Root Mean Square Error
            rmse = torch.sqrt(F.mse_loss(pred_blendshapes, target_blendshapes))
            metrics["rmse"] = rmse.item()

            # Correlation coefficient (per batch)
            corr_coeffs = []
            for i in range(pred_blendshapes.shape[0]):
                pred_flat = pred_blendshapes[i].flatten()
                target_flat = target_blendshapes[i].flatten()

                corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                if not torch.isnan(corr):
                    corr_coeffs.append(corr.item())

            if corr_coeffs:
                metrics["correlation"] = sum(corr_coeffs) / len(corr_coeffs)
            else:
                metrics["correlation"] = 0.0

            # Active blendshape accuracy (threshold-based)
            threshold = 0.1
            pred_active = (pred_blendshapes > threshold).float()
            target_active = (target_blendshapes > threshold).float()

            # Precision and recall for active blendshapes
            tp = (pred_active * target_active).sum()
            fp = (pred_active * (1 - target_active)).sum()
            fn = ((1 - pred_active) * target_active).sum()

            precision = tp / (tp + fp + self.eps)
            recall = tp / (tp + fn + self.eps)
            f1_score = 2 * precision * recall / (precision + recall + self.eps)

            metrics["precision"] = precision.item()
            metrics["recall"] = recall.item()
            metrics["f1_score"] = f1_score.item()

            # Range metrics
            pred_range = pred_blendshapes.max() - pred_blendshapes.min()
            target_range = target_blendshapes.max() - target_blendshapes.min()
            metrics["range_ratio"] = (pred_range / (target_range + self.eps)).item()

        return metrics


class PerceptualBlendshapeLoss(nn.Module):
    """
    Perceptual loss for blendshape consistency.

    Computes loss based on perceptually important blendshape combinations
    and facial landmark consistency.
    """

    def __init__(self):
        super().__init__()

        # Define perceptually important blendshape groups
        self.mouth_group = list(range(12, 32))  # Mouth-related blendshapes
        self.eye_group = list(range(0, 12))  # Eye-related blendshapes
        self.brow_group = list(range(32, 44))  # Brow-related blendshapes
        self.jaw_group = list(range(44, 52))  # Jaw-related blendshapes

        self.groups = {
            "mouth": self.mouth_group,
            "eye": self.eye_group,
            "brow": self.brow_group,
            "jaw": self.jaw_group,
        }

    def forward(
        self,
        pred_blendshapes: torch.Tensor,
        target_blendshapes: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute perceptual loss."""
        total_loss = 0.0

        # Group-wise losses with different weights
        group_weights = {"mouth": 2.0, "eye": 1.0, "brow": 1.0, "jaw": 1.5}

        group_losses = []
        for group_name, indices in self.groups.items():
            pred_group = pred_blendshapes[:, indices]
            target_group = target_blendshapes[:, indices]

            # Weighted MSE for this group
            group_loss = F.mse_loss(pred_group, target_group)
            group_losses.append(group_weights[group_name] * group_loss)

        # Sum all group losses (avoiding in-place operations)
        total_loss = sum(group_losses)

        # Audio-visual consistency loss (if audio features available)
        if audio_features is not None:
            av_loss = self._compute_audiovisual_loss(pred_blendshapes, audio_features)
            total_loss = total_loss + 0.5 * av_loss

        return total_loss

    def _compute_audiovisual_loss(
        self,
        blendshapes: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute audio-visual consistency loss."""
        # Simple correlation-based loss between mouth blendshapes and audio energy
        mouth_activation = blendshapes[:, self.mouth_group].mean(
            dim=1
        )  # Average mouth activation

        if audio_features.dim() == 3:  # (B, T, D)
            audio_energy = audio_features.norm(dim=2).mean(
                dim=1
            )  # Average energy per batch
        else:
            audio_energy = audio_features.norm(dim=1)

        # Normalize both signals
        mouth_norm = F.normalize(mouth_activation.unsqueeze(0), dim=1).squeeze(0)
        audio_norm = F.normalize(audio_energy.unsqueeze(0), dim=1).squeeze(0)

        # Negative correlation as loss (we want positive correlation)
        correlation = F.cosine_similarity(
            mouth_norm.unsqueeze(0), audio_norm.unsqueeze(0)
        )
        av_loss = 1 - correlation.mean()

        return av_loss


class LandmarkConsistencyLoss(nn.Module):
    """
    Loss based on facial landmark consistency.

    Converts blendshapes to approximate landmark positions and computes
    landmark-based loss for better geometric consistency.
    """

    def __init__(self, num_landmarks: int = 68):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Simplified blendshape-to-landmark mapping (would need proper calibration)
        # This is a placeholder - real implementation would use accurate mapping
        self.bs_to_landmark_weights = nn.Parameter(
            torch.randn(num_landmarks * 2, 52) * 0.01,  # 68 landmarks * 2D = 136
            requires_grad=False,
        )

    def forward(
        self,
        pred_blendshapes: torch.Tensor,
        target_blendshapes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute landmark consistency loss."""
        # Convert blendshapes to approximate landmarks
        pred_landmarks = torch.matmul(pred_blendshapes, self.bs_to_landmark_weights.T)
        target_landmarks = torch.matmul(
            target_blendshapes, self.bs_to_landmark_weights.T
        )

        # Reshape to (B, num_landmarks, 2)
        pred_landmarks = pred_landmarks.view(-1, self.num_landmarks, 2)
        target_landmarks = target_landmarks.view(-1, self.num_landmarks, 2)

        # L2 loss on landmark positions
        landmark_loss = F.mse_loss(pred_landmarks, target_landmarks)

        return landmark_loss


class BlendshapeMetrics:
    """
    Comprehensive metrics for blendshape evaluation.

    Provides various metrics for evaluating blendshape generation quality
    including accuracy, smoothness, and perceptual measures.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.audio_features = []

    def update(
        self,
        pred_blendshapes: torch.Tensor,
        target_blendshapes: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ):
        """Update metrics with new batch."""
        self.predictions.append(pred_blendshapes.detach().cpu())
        self.targets.append(target_blendshapes.detach().cpu())

        if audio_features is not None:
            self.audio_features.append(audio_features.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute accumulated metrics."""
        if not self.predictions:
            return {}

        # Concatenate all batches
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        metrics = {}

        # Basic metrics
        metrics["mae"] = F.l1_loss(all_preds, all_targets).item()
        metrics["mse"] = F.mse_loss(all_preds, all_targets).item()
        metrics["rmse"] = torch.sqrt(F.mse_loss(all_preds, all_targets)).item()

        # Per-blendshape metrics
        per_bs_mae = F.l1_loss(all_preds, all_targets, reduction="none").mean(dim=0)
        metrics["max_bs_mae"] = per_bs_mae.max().item()
        metrics["min_bs_mae"] = per_bs_mae.min().item()
        metrics["std_bs_mae"] = per_bs_mae.std().item()

        # Correlation metrics
        correlations = []
        for i in range(52):  # For each blendshape
            pred_bs = all_preds[:, i]
            target_bs = all_targets[:, i]

            if pred_bs.std() > 1e-6 and target_bs.std() > 1e-6:
                corr = torch.corrcoef(torch.stack([pred_bs, target_bs]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.item())

        if correlations:
            metrics["mean_correlation"] = sum(correlations) / len(correlations)
            metrics["min_correlation"] = min(correlations)
        else:
            metrics["mean_correlation"] = 0.0
            metrics["min_correlation"] = 0.0

        # Temporal smoothness (if we have sequences)
        if all_preds.shape[0] > 1:
            pred_diff = torch.diff(all_preds, dim=0)
            target_diff = torch.diff(all_targets, dim=0)

            metrics["temporal_consistency"] = F.l1_loss(pred_diff, target_diff).item()
            metrics["pred_smoothness"] = pred_diff.abs().mean().item()
            metrics["target_smoothness"] = target_diff.abs().mean().item()

        # Activity metrics
        threshold = 0.1
        pred_active = (all_preds > threshold).float()
        target_active = (all_targets > threshold).float()

        metrics["pred_activity"] = pred_active.mean().item()
        metrics["target_activity"] = target_active.mean().item()

        # Precision/Recall for active blendshapes
        tp = (pred_active * target_active).sum()
        fp = (pred_active * (1 - target_active)).sum()
        fn = ((1 - pred_active) * target_active).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        metrics["precision"] = precision.item()
        metrics["recall"] = recall.item()
        metrics["f1_score"] = f1_score.item()

        return metrics


def compute_lip_sync_metrics(
    pred_blendshapes: torch.Tensor,
    target_blendshapes: torch.Tensor,
    audio_features: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute lip synchronization metrics.

    Args:
        pred_blendshapes: Predicted blendshapes (B, 52) or (T, 52)
        target_blendshapes: Target blendshapes (B, 52) or (T, 52)
        audio_features: Optional audio features for audio-visual metrics

    Returns:
        Dictionary of lip sync metrics
    """
    metrics = {}

    # Mouth-related blendshapes (ARKit indices)
    mouth_indices = list(range(12, 32))  # Simplified mouth region

    pred_mouth = pred_blendshapes[:, mouth_indices]
    target_mouth = target_blendshapes[:, mouth_indices]

    # Mouth-specific MAE
    metrics["mouth_mae"] = F.l1_loss(pred_mouth, target_mouth).item()

    # Mouth activity correlation
    pred_mouth_activity = pred_mouth.sum(dim=1)  # Total mouth activation
    target_mouth_activity = target_mouth.sum(dim=1)

    if pred_mouth_activity.std() > 1e-6 and target_mouth_activity.std() > 1e-6:
        mouth_corr = torch.corrcoef(
            torch.stack([pred_mouth_activity, target_mouth_activity])
        )[0, 1]
        metrics["mouth_correlation"] = (
            mouth_corr.item() if not torch.isnan(mouth_corr) else 0.0
        )
    else:
        metrics["mouth_correlation"] = 0.0

    # Audio-visual synchronization (if audio available)
    if audio_features is not None:
        if audio_features.dim() == 3:  # (B, T, D) or (T, B, D)
            audio_energy = audio_features.norm(dim=-1).mean(dim=-1)  # Average energy
        else:
            audio_energy = audio_features.norm(dim=-1)

        # Correlation between mouth activity and audio energy
        if audio_energy.std() > 1e-6:
            av_corr = torch.corrcoef(torch.stack([pred_mouth_activity, audio_energy]))[
                0, 1
            ]
            metrics["audiovisual_sync"] = (
                av_corr.item() if not torch.isnan(av_corr) else 0.0
            )
        else:
            metrics["audiovisual_sync"] = 0.0

    return metrics
