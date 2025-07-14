"""Tests for model.gaussian_face module."""

import pytest
import torch

from src.model.gaussian_face import KoeMorphModel, create_koemorph_model


class TestKoeMorphModel:
    """Test complete KoeMorph model."""
    
    def test_model_creation(self):
        """Test model creation with default parameters."""
        model = KoeMorphModel()
        
        assert model.mel_dim == 80
        assert model.prosody_dim == 4
        assert model.emotion_dim == 256
        assert model.num_blendshapes == 52
        assert model.use_temporal_smoothing is True
        assert model.use_constraints is True
    
    def test_model_creation_custom_params(self):
        """Test model creation with custom parameters."""
        model = KoeMorphModel(
            mel_dim=40,
            prosody_dim=6,
            emotion_dim=128,
            d_model=128,
            num_heads=4,
            num_blendshapes=30,
            use_temporal_smoothing=False,
            use_constraints=False,
        )
        
        assert model.mel_dim == 40
        assert model.prosody_dim == 6
        assert model.emotion_dim == 128
        assert model.d_model == 128
        assert model.num_blendshapes == 30
        assert model.use_temporal_smoothing is False
        assert model.use_constraints is False
    
    def test_model_forward_basic(self):
        """Test basic forward pass."""
        model = KoeMorphModel(
            mel_dim=40,
            prosody_dim=4,
            emotion_dim=64,
            d_model=128,
            num_heads=4,
            num_encoder_layers=1,
            num_attention_layers=2,
        )
        
        batch_size = 2
        seq_len = 15
        
        mel_features = torch.randn(batch_size, seq_len, 40)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 64)
        
        output = model(mel_features, prosody_features, emotion_features)
        
        # Check output structure
        assert 'blendshapes' in output
        assert 'raw_blendshapes' in output
        
        # Check shapes
        assert output['blendshapes'].shape == (batch_size, 52)
        assert output['raw_blendshapes'].shape == (batch_size, 52)
        
        # Check value range (should be in [0,1] for sigmoid activation)
        assert torch.all(output['blendshapes'] >= 0)
        assert torch.all(output['blendshapes'] <= 1)
    
    def test_model_forward_with_mask(self):
        """Test forward pass with audio mask."""
        model = KoeMorphModel(d_model=64, num_heads=2)
        
        batch_size = 2
        seq_len = 10
        
        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)
        
        # Create audio mask
        audio_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        audio_mask[0, 7:] = False  # Mask last 3 frames for first sample
        audio_mask[1, 5:] = False  # Mask last 5 frames for second sample
        
        output = model(
            mel_features, prosody_features, emotion_features, audio_mask=audio_mask
        )
        
        assert output['blendshapes'].shape == (batch_size, 52)
    
    def test_model_forward_with_prev_blendshapes(self):
        """Test forward pass with previous blendshapes."""
        model = KoeMorphModel(d_model=64, num_heads=2)
        
        batch_size = 1
        seq_len = 8
        
        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)
        prev_blendshapes = torch.rand(batch_size, 52)
        
        output = model(
            mel_features, prosody_features, emotion_features,
            prev_blendshapes=prev_blendshapes
        )
        
        # Output should be influenced by previous state
        output_no_prev = model(mel_features, prosody_features, emotion_features)
        
        # Should be different (though this is probabilistic)
        assert not torch.allclose(output['blendshapes'], output_no_prev['blendshapes'])
    
    def test_model_forward_with_attention_return(self):
        """Test forward pass with attention weights return."""
        model = KoeMorphModel(
            d_model=64, num_heads=2, num_attention_layers=2
        )
        
        batch_size = 1
        seq_len = 5
        
        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)
        
        output = model(
            mel_features, prosody_features, emotion_features,
            return_attention=True
        )
        
        assert 'attention_weights' in output
        assert len(output['attention_weights']) == 2  # num_attention_layers
        
        # Check attention weights shape
        for attn_weights in output['attention_weights']:
            assert attn_weights.shape == (batch_size, 2, 52, seq_len)  # (B, H, Q, T)
    
    def test_model_without_smoothing_constraints(self):
        """Test model without smoothing and constraints."""
        model = KoeMorphModel(
            d_model=64,
            use_temporal_smoothing=False,
            use_constraints=False,
        )
        
        batch_size = 1
        seq_len = 5
        
        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)
        
        output = model(
            mel_features, prosody_features, emotion_features,
            apply_smoothing=False,
            apply_constraints=False
        )
        
        # Raw and final blendshapes should be the same
        assert torch.allclose(output['blendshapes'], output['raw_blendshapes'])
    
    def test_model_inference_step(self):
        """Test single inference step for real-time processing."""
        model = KoeMorphModel(d_model=64, num_heads=2)
        
        # Single frame input
        mel_features = torch.randn(1, 1, 80)
        prosody_features = torch.randn(1, 1, 4)
        emotion_features = torch.randn(1, 1, 256)
        prev_blendshapes = torch.rand(1, 52)
        
        blendshapes = model.inference_step(
            mel_features, prosody_features, emotion_features, prev_blendshapes
        )
        
        assert blendshapes.shape == (1, 52)
        assert torch.all(blendshapes >= 0)
        assert torch.all(blendshapes <= 1)
    
    def test_model_reset_temporal_state(self):
        """Test temporal state reset."""
        model = KoeMorphModel(use_temporal_smoothing=True)
        
        # Process some frames to build up state
        for _ in range(3):
            mel_features = torch.randn(1, 1, 80)
            prosody_features = torch.randn(1, 1, 4)
            emotion_features = torch.randn(1, 1, 256)
            
            model.inference_step(mel_features, prosody_features, emotion_features)
        
        # Reset state
        model.reset_temporal_state()
        
        # State should be reset (hard to verify directly, but should not crash)
        mel_features = torch.randn(1, 1, 80)
        prosody_features = torch.randn(1, 1, 4)
        emotion_features = torch.randn(1, 1, 256)
        
        blendshapes = model.inference_step(mel_features, prosody_features, emotion_features)
        assert blendshapes.shape == (1, 52)
    
    def test_model_gradient_flow(self):
        """Test gradient flow through complete model."""
        model = KoeMorphModel(d_model=64, num_heads=2, num_attention_layers=1)
        
        batch_size = 1
        seq_len = 3
        
        mel_features = torch.randn(batch_size, seq_len, 80, requires_grad=True)
        prosody_features = torch.randn(batch_size, seq_len, 4, requires_grad=True)
        emotion_features = torch.randn(batch_size, seq_len, 256, requires_grad=True)
        
        output = model(mel_features, prosody_features, emotion_features)
        loss = output['blendshapes'].sum()
        loss.backward()
        
        # Check gradients exist for inputs
        assert mel_features.grad is not None
        assert prosody_features.grad is not None
        assert emotion_features.grad is not None
        
        # Check gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_num_parameters(self):
        """Test parameter counting."""
        model = KoeMorphModel(d_model=64, num_heads=2)
        
        num_params = model.get_num_parameters()
        
        assert isinstance(num_params, int)
        assert num_params > 0
        
        # Compare with manual counting
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert num_params == manual_count
    
    def test_model_info(self):
        """Test model info retrieval."""
        model = KoeMorphModel(
            mel_dim=40,
            d_model=128,
            num_heads=4,
            num_attention_layers=3,
        )
        
        info = model.get_model_info()
        
        assert info['mel_dim'] == 40
        assert info['d_model'] == 128
        assert info['num_attention_layers'] == 3
        assert info['use_temporal_smoothing'] is True
        assert 'total_parameters' in info


class TestModelCreation:
    """Test model creation utilities."""
    
    def test_create_from_config(self):
        """Test model creation from configuration dictionary."""
        config = {
            'mel_dim': 40,
            'prosody_dim': 6,
            'emotion_dim': 128,
            'd_model': 64,
            'num_heads': 2,
            'num_encoder_layers': 1,
            'num_attention_layers': 2,
            'decoder_hidden_dim': 32,
            'use_temporal_smoothing': False,
            'use_constraints': False,
        }
        
        model = create_koemorph_model(config)
        
        assert model.mel_dim == 40
        assert model.prosody_dim == 6
        assert model.emotion_dim == 128
        assert model.d_model == 64
        assert model.use_temporal_smoothing is False
        assert model.use_constraints is False
    
    def test_create_from_partial_config(self):
        """Test model creation with partial configuration (uses defaults)."""
        config = {
            'd_model': 128,
            'num_heads': 8,
        }
        
        model = create_koemorph_model(config)
        
        # Should use specified values
        assert model.d_model == 128
        
        # Should use defaults for unspecified values
        assert model.mel_dim == 80  # default
        assert model.prosody_dim == 4  # default
        assert model.num_blendshapes == 52  # default
    
    def test_create_from_empty_config(self):
        """Test model creation with empty configuration (all defaults)."""
        config = {}
        
        model = create_koemorph_model(config)
        
        # Should use all defaults
        assert model.mel_dim == 80
        assert model.prosody_dim == 4
        assert model.emotion_dim == 256
        assert model.d_model == 256
        assert model.num_blendshapes == 52
        assert model.use_temporal_smoothing is True
        assert model.use_constraints is True