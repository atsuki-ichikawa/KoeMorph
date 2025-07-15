"""Tests for model.decoder module."""

import pytest
import torch

from src.model.decoder import (BlendshapeConstraints, BlendshapeDecoder,
                               TemporalSmoother, validate_blendshape_output)


class TestBlendshapeDecoder:
    """Test blendshape decoder functionality."""

    def test_decoder_creation(self):
        """Test decoder creation."""
        decoder = BlendshapeDecoder(
            d_model=256,
            hidden_dim=128,
            num_blendshapes=52,
            num_layers=2,
            activation="gelu",
            output_activation="sigmoid",
        )

        assert decoder.d_model == 256
        assert decoder.hidden_dim == 128
        assert decoder.num_blendshapes == 52
        assert decoder.num_layers == 2
        assert decoder.output_activation == "sigmoid"

    def test_decoder_forward_basic(self):
        """Test basic decoder forward pass."""
        decoder = BlendshapeDecoder(
            d_model=128, hidden_dim=64, num_blendshapes=52, num_layers=1
        )

        batch_size = 2
        attention_output = torch.randn(batch_size, 52, 128)

        blendshapes = decoder(attention_output)

        assert blendshapes.shape == (batch_size, 52)

        # Check output range for sigmoid activation
        assert torch.all(blendshapes >= 0)
        assert torch.all(blendshapes <= 1)

    def test_decoder_with_residual(self):
        """Test decoder with residual connection."""
        decoder = BlendshapeDecoder(
            d_model=64, hidden_dim=32, num_blendshapes=52, use_residual=True
        )

        batch_size = 1
        attention_output = torch.randn(batch_size, 52, 64)
        prev_blendshapes = torch.rand(batch_size, 52)  # Previous state in [0,1]

        blendshapes = decoder(attention_output, prev_blendshapes)

        assert blendshapes.shape == (batch_size, 52)

        # With residual, output should be influenced by previous state
        blendshapes_no_residual = decoder(attention_output, None)
        assert not torch.allclose(blendshapes, blendshapes_no_residual)

    def test_decoder_activations(self):
        """Test different output activations."""
        activations = ["sigmoid", "tanh", "none"]

        for activation in activations:
            decoder = BlendshapeDecoder(
                d_model=64,
                hidden_dim=32,
                num_blendshapes=52,
                output_activation=activation,
            )

            attention_output = torch.randn(1, 52, 64)
            blendshapes = decoder(attention_output)

            assert blendshapes.shape == (1, 52)

            if activation == "sigmoid":
                assert torch.all(blendshapes >= 0)
                assert torch.all(blendshapes <= 1)
            elif activation == "tanh":
                assert torch.all(blendshapes >= -1)
                assert torch.all(blendshapes <= 1)
            # For "none", no specific range constraints

    def test_decoder_layer_norm(self):
        """Test decoder with layer normalization."""
        decoder = BlendshapeDecoder(
            d_model=64,
            hidden_dim=32,
            num_blendshapes=52,
            num_layers=2,
            use_layer_norm=True,
        )

        attention_output = torch.randn(1, 52, 64)
        blendshapes = decoder(attention_output)

        assert blendshapes.shape == (1, 52)

        # Check that layer norms exist
        assert len(decoder.layer_norms) == 2

    def test_decoder_gradient_flow(self):
        """Test gradient flow through decoder."""
        decoder = BlendshapeDecoder(d_model=32, hidden_dim=16, num_blendshapes=52)

        attention_output = torch.randn(1, 52, 32, requires_grad=True)
        blendshapes = decoder(attention_output)
        loss = blendshapes.sum()
        loss.backward()

        # Check gradients exist
        assert attention_output.grad is not None
        assert not torch.all(attention_output.grad == 0)

    def test_decoder_invalid_input_shape(self):
        """Test error handling for invalid input shape."""
        decoder = BlendshapeDecoder(d_model=64, num_blendshapes=52)

        # Wrong number of blendshapes
        invalid_input = torch.randn(1, 30, 64)  # 30 instead of 52

        with pytest.raises(ValueError, match="Expected 52 blendshapes"):
            decoder(invalid_input)

    def test_decoder_custom_activation(self):
        """Test decoder with different activation functions."""
        activations = ["relu", "gelu", "swish", "leaky_relu"]

        for activation in activations:
            decoder = BlendshapeDecoder(
                d_model=32, hidden_dim=16, num_blendshapes=52, activation=activation
            )

            attention_output = torch.randn(1, 52, 32)
            blendshapes = decoder(attention_output)

            assert blendshapes.shape == (1, 52)

    def test_decoder_invalid_activation(self):
        """Test error handling for invalid activation."""
        with pytest.raises(ValueError, match="Unknown activation"):
            BlendshapeDecoder(activation="invalid_activation")

    def test_decoder_no_bias(self):
        """Test decoder without bias terms."""
        decoder = BlendshapeDecoder(
            d_model=32, hidden_dim=16, num_blendshapes=52, bias=False
        )

        # Check that linear layers have no bias
        assert decoder.input_proj.bias is None
        assert decoder.output_proj.bias is None
        for layer in decoder.hidden_layers:
            assert layer.bias is None


class TestTemporalSmoother:
    """Test temporal smoothing functionality."""

    def test_smoother_creation(self):
        """Test smoother creation."""
        smoother = TemporalSmoother(
            num_blendshapes=52, smoothing_method="exponential", alpha=0.8, window_size=5
        )

        assert smoother.num_blendshapes == 52
        assert smoother.smoothing_method == "exponential"
        assert smoother.window_size == 5

    def test_exponential_smoothing(self):
        """Test exponential smoothing."""
        smoother = TemporalSmoother(
            num_blendshapes=52,
            smoothing_method="exponential",
            alpha=0.8,
            learnable=False,
        )

        batch_size = 1
        blendshapes1 = torch.ones(batch_size, 52) * 0.5
        blendshapes2 = torch.ones(batch_size, 52) * 1.0

        # Reset state first
        smoother.forward(blendshapes1, reset_state=True)

        # First call
        smooth1 = smoother(blendshapes1)
        assert smooth1.shape == (batch_size, 52)

        # Second call should be smoothed
        smooth2 = smoother(blendshapes2)

        # Should be between previous output and current input
        assert torch.all(smooth2 > smooth1)
        assert torch.all(smooth2 < blendshapes2)

    def test_gaussian_smoothing(self):
        """Test Gaussian smoothing."""
        smoother = TemporalSmoother(
            num_blendshapes=52,
            smoothing_method="gaussian",
            window_size=3,
            learnable=False,
        )

        batch_size = 1

        # Reset and feed several frames
        blendshapes = torch.rand(batch_size, 52)
        smooth = smoother(blendshapes, reset_state=True)

        for _ in range(5):
            blendshapes = torch.rand(batch_size, 52)
            smooth = smoother(blendshapes)
            assert smooth.shape == (batch_size, 52)

    def test_median_smoothing(self):
        """Test median smoothing."""
        smoother = TemporalSmoother(
            num_blendshapes=52, smoothing_method="median", window_size=5
        )

        batch_size = 2

        # Reset and feed several frames
        blendshapes = torch.rand(batch_size, 52)
        smooth = smoother(blendshapes, reset_state=True)

        for _ in range(10):
            blendshapes = torch.rand(batch_size, 52)
            smooth = smoother(blendshapes)
            assert smooth.shape == (batch_size, 52)

    def test_learnable_smoothing(self):
        """Test learnable smoothing parameters."""
        smoother = TemporalSmoother(
            num_blendshapes=52,
            smoothing_method="exponential",
            alpha=0.5,
            learnable=True,
        )

        # Check that alpha is a parameter
        assert isinstance(smoother.alpha, torch.nn.Parameter)

        batch_size = 1
        blendshapes = torch.rand(batch_size, 52)
        smooth = smoother(blendshapes, reset_state=True)

        # Test gradient flow
        loss = smooth.sum()
        loss.backward()

        assert smoother.alpha.grad is not None

    def test_batch_size_changes(self):
        """Test handling of changing batch sizes."""
        smoother = TemporalSmoother(num_blendshapes=52, smoothing_method="exponential")

        # Start with batch size 1
        blendshapes1 = torch.rand(1, 52)
        smooth1 = smoother(blendshapes1, reset_state=True)

        # Change to batch size 3
        blendshapes2 = torch.rand(3, 52)
        smooth2 = smoother(blendshapes2)

        assert smooth2.shape == (3, 52)

    def test_invalid_smoothing_method(self):
        """Test error handling for invalid smoothing method."""
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smoother = TemporalSmoother(smoothing_method="invalid")
            blendshapes = torch.rand(1, 52)
            smoother(blendshapes)


class TestBlendshapeConstraints:
    """Test blendshape constraint functionality."""

    def test_constraints_creation(self):
        """Test constraints creation."""
        constraints = BlendshapeConstraints(
            num_blendshapes=52,
            mutual_exclusions=[(0, 1), (2, 3)],
            smoothness_weight=0.1,
        )

        assert constraints.num_blendshapes == 52
        assert len(constraints.exclusion_pairs) == 2
        assert constraints.smoothness_weight == 0.1

    def test_value_range_constraints(self):
        """Test value range constraints."""
        # Create constraints with custom value ranges
        value_constraints = {
            0: (0.1, 0.9),  # Blendshape 0: range [0.1, 0.9]
            1: (0.0, 0.5),  # Blendshape 1: range [0.0, 0.5]
        }

        constraints = BlendshapeConstraints(
            num_blendshapes=52, value_constraints=value_constraints
        )

        # Test with values outside range
        blendshapes = torch.ones(1, 52) * 0.8
        blendshapes[0, 0] = 1.2  # Above max for bs 0
        blendshapes[0, 1] = 0.7  # Above max for bs 1

        constrained, violations = constraints(blendshapes, return_violations=True)

        # Should be clamped to valid ranges
        assert constrained[0, 0] <= 0.9
        assert constrained[0, 1] <= 0.5

        # Should report violations
        assert violations["range_violations"] > 0

    def test_mutual_exclusion_constraints(self):
        """Test mutual exclusion constraints."""
        constraints = BlendshapeConstraints(
            num_blendshapes=52,
            mutual_exclusions=[(0, 1)],  # Blendshapes 0 and 1 are mutually exclusive
        )

        # Test with both blendshapes active
        blendshapes = torch.zeros(1, 52)
        blendshapes[0, 0] = 0.8
        blendshapes[0, 1] = 0.6

        constrained, violations = constraints(blendshapes, return_violations=True)

        # Should reduce both values due to mutual exclusion
        assert constrained[0, 0] < 0.8
        assert constrained[0, 1] < 0.6

        # Should report violations
        assert violations["exclusion_0_1"] > 0

    def test_temporal_smoothness(self):
        """Test temporal smoothness constraint."""
        constraints = BlendshapeConstraints(num_blendshapes=52)

        # First call
        blendshapes1 = torch.zeros(1, 52)
        constrained1, violations1 = constraints(blendshapes1, return_violations=True)

        # Second call with different values
        blendshapes2 = torch.ones(1, 52)
        constrained2, violations2 = constraints(blendshapes2, return_violations=True)

        # Should report temporal smoothness violation
        assert violations2["temporal_smoothness"] > 0

    def test_constraints_without_application(self):
        """Test constraint violation detection without applying constraints."""
        constraints = BlendshapeConstraints(num_blendshapes=52)

        # Create blendshapes with violations
        blendshapes = torch.ones(1, 52) * 1.5  # Above valid range

        constrained, violations = constraints(
            blendshapes, apply_constraints=False, return_violations=True
        )

        # Values should be unchanged
        assert torch.allclose(constrained, blendshapes)

        # But violations should be detected
        assert violations["range_violations"] > 0

    def test_constraints_reset_state(self):
        """Test state reset functionality."""
        constraints = BlendshapeConstraints(num_blendshapes=52)

        # Set some state
        blendshapes = torch.rand(1, 52)
        constraints(blendshapes)

        # Reset state
        constraints.reset_state()

        # Previous state should be zero
        assert torch.allclose(constraints.prev_blendshapes, torch.zeros(1, 52))


class TestBlendshapeValidation:
    """Test blendshape validation utilities."""

    def test_valid_blendshapes(self):
        """Test validation with valid blendshapes."""
        blendshapes = torch.rand(10, 52)  # Valid range [0,1]

        results = validate_blendshape_output(blendshapes)

        assert results["valid"] is True
        assert "value_range" in results["stats"]
        assert "mean_activation" in results["stats"]
        assert "active_blendshapes" in results["stats"]

    def test_invalid_shape(self):
        """Test validation with invalid shape."""
        invalid_blendshapes = torch.rand(10, 30)  # Wrong number of blendshapes

        results = validate_blendshape_output(invalid_blendshapes)

        assert results["valid"] is False
        assert any("Expected shape" in w for w in results["warnings"])

    def test_out_of_range_values(self):
        """Test validation with out-of-range values."""
        blendshapes = torch.rand(5, 52)
        blendshapes[0, 0] = -0.5  # Negative value
        blendshapes[1, 1] = 1.5  # Above 1.0

        results = validate_blendshape_output(blendshapes)

        assert any("Negative values" in w for w in results["warnings"])
        assert any("Values above 1" in w for w in results["warnings"])

    def test_nan_detection(self):
        """Test detection of NaN values."""
        blendshapes = torch.rand(5, 52)
        blendshapes[0, 0] = float("nan")

        results = validate_blendshape_output(blendshapes)

        assert results["valid"] is False
        assert any("NaN values" in w for w in results["warnings"])

    def test_inf_detection(self):
        """Test detection of infinite values."""
        blendshapes = torch.rand(5, 52)
        blendshapes[1, 10] = float("inf")

        results = validate_blendshape_output(blendshapes)

        assert results["valid"] is False
        assert any("Infinite values" in w for w in results["warnings"])

    def test_dead_blendshapes_warning(self):
        """Test warning for dead (inactive) blendshapes."""
        blendshapes = torch.zeros(10, 52)  # All blendshapes inactive
        blendshapes[:, :5] = torch.rand(10, 5) * 0.5  # Only first 5 active

        results = validate_blendshape_output(blendshapes)

        assert results["stats"]["dead_blendshapes"] == 47  # 52 - 5 = 47 dead
        assert any("inactive blendshapes" in w for w in results["warnings"])

    def test_saturated_blendshapes_warning(self):
        """Test warning for saturated blendshapes."""
        blendshapes = torch.ones(10, 52) * 0.95  # Most blendshapes near max

        results = validate_blendshape_output(blendshapes)

        # Should detect many saturated blendshapes
        assert results["stats"]["saturated_blendshapes"] > 0
        assert any("saturated blendshapes" in w for w in results["warnings"])

    def test_statistics_computation(self):
        """Test that statistics are computed correctly."""
        # Create controlled blendshapes
        blendshapes = torch.zeros(5, 52)
        blendshapes[:, :10] = 0.5  # First 10 blendshapes at 0.5
        blendshapes[:, 10:20] = 0.2  # Next 10 at 0.2
        # Rest remain at 0

        results = validate_blendshape_output(blendshapes)

        # Check statistics
        assert results["stats"]["active_blendshapes"] == 10  # Only first 10 > 0.1
        assert results["stats"]["dead_blendshapes"] == 32  # 52 - 20 = 32 dead

        # Mean activation should be weighted average
        expected_mean = (10 * 0.5 + 10 * 0.2) / 52
        assert abs(results["stats"]["mean_activation"] - expected_mean) < 0.01
