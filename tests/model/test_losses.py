"""Tests for model.losses module."""

import pytest
import torch

from src.model.losses import (BlendshapeMetrics, KoeMorphLoss,
                              LandmarkConsistencyLoss,
                              PerceptualBlendshapeLoss,
                              compute_lip_sync_metrics)


class TestKoeMorphLoss:
    """Test combined loss function."""

    def test_loss_creation(self):
        """Test loss function creation."""
        loss_fn = KoeMorphLoss(
            mse_weight=1.0, l1_weight=0.1, perceptual_weight=0.5, temporal_weight=0.2
        )

        assert loss_fn.mse_weight == 1.0
        assert loss_fn.l1_weight == 0.1
        assert loss_fn.perceptual_weight == 0.5
        assert loss_fn.temporal_weight == 0.2

    def test_basic_reconstruction_loss(self):
        """Test basic MSE and L1 reconstruction losses."""
        loss_fn = KoeMorphLoss(
            mse_weight=1.0,
            l1_weight=0.1,
            perceptual_weight=0.0,  # Disable other losses
            temporal_weight=0.0,
            sparsity_weight=0.0,
            smoothness_weight=0.0,
            landmark_weight=0.0,
        )

        batch_size = 4
        pred_blendshapes = torch.rand(batch_size, 52)
        target_blendshapes = torch.rand(batch_size, 52)

        total_loss, metrics = loss_fn(pred_blendshapes, target_blendshapes)

        # Check that loss is computed
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() > 0

        # Check that metrics are returned
        assert "mse" in metrics
        assert "l1" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_temporal_loss(self):
        """Test temporal consistency loss."""
        loss_fn = KoeMorphLoss(
            mse_weight=0.0, l1_weight=0.0, temporal_weight=1.0, velocity_weight=1.0
        )

        batch_size = 2
        pred_current = torch.rand(batch_size, 52)
        target_current = torch.rand(batch_size, 52)
        pred_prev = torch.rand(batch_size, 52)
        target_prev = torch.rand(batch_size, 52)

        total_loss, metrics = loss_fn(
            pred_current, target_current, pred_prev, target_prev
        )

        assert "temporal" in metrics
        assert "velocity" in metrics
        assert total_loss.item() > 0

    def test_regularization_losses(self):
        """Test sparsity and smoothness regularization."""
        loss_fn = KoeMorphLoss(
            mse_weight=0.0, l1_weight=0.0, sparsity_weight=0.1, smoothness_weight=0.1
        )

        batch_size = 2
        pred_blendshapes = torch.rand(batch_size, 52)
        target_blendshapes = torch.rand(batch_size, 52)

        total_loss, metrics = loss_fn(pred_blendshapes, target_blendshapes)

        assert "sparsity" in metrics
        assert "smoothness" in metrics
        assert total_loss.item() > 0

    def test_identical_predictions(self):
        """Test loss with identical predictions and targets."""
        loss_fn = KoeMorphLoss(
            mse_weight=1.0,
            l1_weight=1.0,
            perceptual_weight=0.0,
            temporal_weight=0.0,
            sparsity_weight=0.0,
            smoothness_weight=0.0,
        )

        batch_size = 2
        blendshapes = torch.rand(batch_size, 52)

        total_loss, metrics = loss_fn(blendshapes, blendshapes)

        # MSE and L1 should be near zero for identical inputs
        assert metrics["mse"] < 1e-6
        assert metrics["l1"] < 1e-6
        assert total_loss.item() < 1e-5

    def test_gradient_flow(self):
        """Test gradient flow through loss function."""
        loss_fn = KoeMorphLoss()

        pred_blendshapes = torch.rand(2, 52, requires_grad=True)
        target_blendshapes = torch.rand(2, 52)

        total_loss, _ = loss_fn(pred_blendshapes, target_blendshapes)
        total_loss.backward()

        # Check gradients exist
        assert pred_blendshapes.grad is not None
        assert not torch.all(pred_blendshapes.grad == 0)

    def test_metrics_computation(self):
        """Test additional metrics computation."""
        loss_fn = KoeMorphLoss()

        # Create controlled predictions and targets
        pred_blendshapes = torch.zeros(3, 52)
        pred_blendshapes[:, :10] = 0.8  # First 10 blendshapes active

        target_blendshapes = torch.zeros(3, 52)
        target_blendshapes[:, :8] = 0.9  # First 8 blendshapes active
        target_blendshapes[:, 8:12] = 0.1  # Next 4 slightly active

        _, metrics = loss_fn(pred_blendshapes, target_blendshapes)

        # Check that all expected metrics are present
        expected_metrics = [
            "mae",
            "rmse",
            "correlation",
            "precision",
            "recall",
            "f1_score",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)

    def test_loss_reduction_modes(self):
        """Test different reduction modes."""
        reductions = ["mean", "sum"]

        for reduction in reductions:
            loss_fn = KoeMorphLoss(reduction=reduction)

            pred_blendshapes = torch.rand(2, 52)
            target_blendshapes = torch.rand(2, 52)

            total_loss, _ = loss_fn(pred_blendshapes, target_blendshapes)

            assert isinstance(total_loss, torch.Tensor)
            assert total_loss.numel() == 1  # Should be scalar


class TestPerceptualBlendshapeLoss:
    """Test perceptual loss component."""

    def test_perceptual_loss_creation(self):
        """Test perceptual loss creation."""
        perceptual_loss = PerceptualBlendshapeLoss()

        # Check that blendshape groups are defined
        assert hasattr(perceptual_loss, "groups")
        assert "mouth" in perceptual_loss.groups
        assert "eye" in perceptual_loss.groups
        assert "brow" in perceptual_loss.groups
        assert "jaw" in perceptual_loss.groups

    def test_perceptual_loss_forward(self):
        """Test perceptual loss forward pass."""
        perceptual_loss = PerceptualBlendshapeLoss()

        batch_size = 2
        pred_blendshapes = torch.rand(batch_size, 52)
        target_blendshapes = torch.rand(batch_size, 52)

        loss = perceptual_loss(pred_blendshapes, target_blendshapes)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_perceptual_loss_with_audio(self):
        """Test perceptual loss with audio features."""
        perceptual_loss = PerceptualBlendshapeLoss()

        batch_size = 2
        pred_blendshapes = torch.rand(batch_size, 52)
        target_blendshapes = torch.rand(batch_size, 52)
        audio_features = torch.randn(batch_size, 30, 256)  # Audio features

        loss = perceptual_loss(pred_blendshapes, target_blendshapes, audio_features)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_audiovisual_consistency(self):
        """Test audio-visual consistency loss."""
        perceptual_loss = PerceptualBlendshapeLoss()

        batch_size = 3
        # Create mouth blendshapes that correlate with audio
        blendshapes = torch.zeros(batch_size, 52)
        blendshapes[:, perceptual_loss.mouth_group] = torch.rand(
            batch_size, len(perceptual_loss.mouth_group)
        )

        # Create audio features with similar pattern
        audio_features = torch.randn(batch_size, 20, 128)

        loss = perceptual_loss(blendshapes, blendshapes, audio_features)

        # Loss should be computed without errors
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_perceptual_loss_gradient_flow(self):
        """Test gradient flow through perceptual loss."""
        perceptual_loss = PerceptualBlendshapeLoss()

        pred_blendshapes = torch.rand(1, 52, requires_grad=True)
        target_blendshapes = torch.rand(1, 52)

        loss = perceptual_loss(pred_blendshapes, target_blendshapes)
        loss.backward()

        assert pred_blendshapes.grad is not None
        assert not torch.all(pred_blendshapes.grad == 0)


class TestLandmarkConsistencyLoss:
    """Test landmark consistency loss."""

    def test_landmark_loss_creation(self):
        """Test landmark loss creation."""
        landmark_loss = LandmarkConsistencyLoss(num_landmarks=68)

        assert landmark_loss.num_landmarks == 68
        assert landmark_loss.bs_to_landmark_weights.shape == (136, 52)  # 68 * 2, 52

    def test_landmark_loss_forward(self):
        """Test landmark loss forward pass."""
        landmark_loss = LandmarkConsistencyLoss()

        batch_size = 2
        pred_blendshapes = torch.rand(batch_size, 52)
        target_blendshapes = torch.rand(batch_size, 52)

        loss = landmark_loss(pred_blendshapes, target_blendshapes)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_landmark_loss_identical_inputs(self):
        """Test landmark loss with identical inputs."""
        landmark_loss = LandmarkConsistencyLoss()

        batch_size = 1
        blendshapes = torch.rand(batch_size, 52)

        loss = landmark_loss(blendshapes, blendshapes)

        # Should be very small for identical inputs
        assert loss.item() < 1e-5

    def test_landmark_loss_gradient_flow(self):
        """Test gradient flow through landmark loss."""
        landmark_loss = LandmarkConsistencyLoss()

        pred_blendshapes = torch.rand(1, 52, requires_grad=True)
        target_blendshapes = torch.rand(1, 52)

        loss = landmark_loss(pred_blendshapes, target_blendshapes)
        loss.backward()

        assert pred_blendshapes.grad is not None


class TestBlendshapeMetrics:
    """Test comprehensive metrics computation."""

    def test_metrics_creation(self):
        """Test metrics object creation."""
        metrics = BlendshapeMetrics()

        assert hasattr(metrics, "predictions")
        assert hasattr(metrics, "targets")
        assert len(metrics.predictions) == 0

    def test_metrics_update_and_compute(self):
        """Test metrics update and computation."""
        metrics = BlendshapeMetrics()

        # Add several batches
        for _ in range(3):
            pred = torch.rand(4, 52)
            target = torch.rand(4, 52)
            metrics.update(pred, target)

        # Compute metrics
        computed_metrics = metrics.compute()

        # Check that metrics are computed
        expected_metrics = [
            "mae",
            "mse",
            "rmse",
            "max_bs_mae",
            "min_bs_mae",
            "mean_correlation",
            "precision",
            "recall",
            "f1_score",
        ]

        for metric in expected_metrics:
            assert metric in computed_metrics
            assert isinstance(computed_metrics[metric], float)

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = BlendshapeMetrics()

        # Add some data
        pred = torch.rand(2, 52)
        target = torch.rand(2, 52)
        metrics.update(pred, target)

        assert len(metrics.predictions) == 1

        # Reset
        metrics.reset()

        assert len(metrics.predictions) == 0
        assert len(metrics.targets) == 0

    def test_metrics_with_audio(self):
        """Test metrics with audio features."""
        metrics = BlendshapeMetrics()

        pred = torch.rand(2, 52)
        target = torch.rand(2, 52)
        audio = torch.randn(2, 30, 128)

        metrics.update(pred, target, audio)

        assert len(metrics.audio_features) == 1

    def test_metrics_empty_computation(self):
        """Test metrics computation with no data."""
        metrics = BlendshapeMetrics()

        computed_metrics = metrics.compute()

        # Should return empty dict
        assert computed_metrics == {}

    def test_metrics_temporal_analysis(self):
        """Test temporal metrics computation."""
        metrics = BlendshapeMetrics()

        # Add sequential data to simulate temporal sequence
        for i in range(10):
            pred = torch.rand(1, 52) * (i + 1) / 10  # Gradually increasing
            target = torch.rand(1, 52) * (i + 1) / 10
            metrics.update(pred, target)

        computed_metrics = metrics.compute()

        # Should include temporal metrics
        assert "temporal_consistency" in computed_metrics
        assert "pred_smoothness" in computed_metrics
        assert "target_smoothness" in computed_metrics

    def test_metrics_correlation_edge_cases(self):
        """Test correlation computation with edge cases."""
        metrics = BlendshapeMetrics()

        # Constant predictions (zero variance)
        pred = torch.ones(5, 52) * 0.5
        target = torch.rand(5, 52)
        metrics.update(pred, target)

        computed_metrics = metrics.compute()

        # Should handle zero variance gracefully
        assert "mean_correlation" in computed_metrics
        assert not torch.isnan(torch.tensor(computed_metrics["mean_correlation"]))


class TestLipSyncMetrics:
    """Test lip synchronization metrics."""

    def test_lip_sync_basic(self):
        """Test basic lip sync metrics computation."""
        pred_blendshapes = torch.rand(10, 52)
        target_blendshapes = torch.rand(10, 52)

        metrics = compute_lip_sync_metrics(pred_blendshapes, target_blendshapes)

        assert "mouth_mae" in metrics
        assert "mouth_correlation" in metrics
        assert isinstance(metrics["mouth_mae"], float)
        assert isinstance(metrics["mouth_correlation"], float)

    def test_lip_sync_with_audio(self):
        """Test lip sync metrics with audio features."""
        pred_blendshapes = torch.rand(10, 52)
        target_blendshapes = torch.rand(10, 52)
        audio_features = torch.randn(10, 128)

        metrics = compute_lip_sync_metrics(
            pred_blendshapes, target_blendshapes, audio_features
        )

        assert "mouth_mae" in metrics
        assert "mouth_correlation" in metrics
        assert "audiovisual_sync" in metrics

    def test_lip_sync_perfect_correlation(self):
        """Test lip sync metrics with perfect correlation."""
        # Create perfectly correlated mouth movements
        mouth_indices = list(range(12, 32))

        pred_blendshapes = torch.zeros(10, 52)
        target_blendshapes = torch.zeros(10, 52)

        # Same mouth pattern for both
        mouth_pattern = torch.rand(10, len(mouth_indices))
        pred_blendshapes[:, mouth_indices] = mouth_pattern
        target_blendshapes[:, mouth_indices] = mouth_pattern

        metrics = compute_lip_sync_metrics(pred_blendshapes, target_blendshapes)

        # Should have perfect correlation
        assert abs(metrics["mouth_correlation"] - 1.0) < 1e-5
        assert metrics["mouth_mae"] < 1e-6

    def test_lip_sync_no_variation(self):
        """Test lip sync metrics with no variation."""
        # Constant blendshapes (no variation)
        pred_blendshapes = torch.ones(10, 52) * 0.5
        target_blendshapes = torch.ones(10, 52) * 0.3

        metrics = compute_lip_sync_metrics(pred_blendshapes, target_blendshapes)

        # Should handle zero variance gracefully
        assert metrics["mouth_correlation"] == 0.0
        assert isinstance(metrics["mouth_mae"], float)

    def test_lip_sync_3d_audio_features(self):
        """Test lip sync with 3D audio features."""
        pred_blendshapes = torch.rand(5, 52)
        target_blendshapes = torch.rand(5, 52)
        audio_features = torch.randn(5, 20, 128)  # 3D features

        metrics = compute_lip_sync_metrics(
            pred_blendshapes, target_blendshapes, audio_features
        )

        assert "audiovisual_sync" in metrics
        assert isinstance(metrics["audiovisual_sync"], float)
        assert not torch.isnan(torch.tensor(metrics["audiovisual_sync"]))
