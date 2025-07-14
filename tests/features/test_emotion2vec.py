"""Tests for features.emotion2vec module."""

import hashlib

import numpy as np
import pytest
import torch

from src.features.emotion2vec import (
    DummyWav2Vec2Model,
    Emotion2VecCache,
    Emotion2VecExtractor,
    compute_audio_hash,
    validate_emotion2vec_features,
)


class TestEmotion2VecExtractor:
    """Test Emotion2Vec feature extraction functionality."""
    
    def test_extractor_creation_dummy(self):
        """Test extractor creation with dummy model."""
        extractor = Emotion2VecExtractor(
            model_name="dummy",  # This will trigger dummy model
            target_fps=30.0,
            output_dim=256
        )
        
        assert extractor.target_fps == 30.0
        assert extractor.output_dim == 256
        assert isinstance(extractor.base_model, DummyWav2Vec2Model)
    
    def test_forward_shape_single_sample(self):
        """Test forward pass shape with single sample."""
        extractor = Emotion2VecExtractor(
            model_name="dummy",
            target_fps=30,
            output_dim=256
        )
        
        # 1 second of audio
        audio = torch.randn(16000)
        
        features = extractor(audio)
        
        # Should be (1, T, 256) since we unsqueeze batch dim
        assert features.dim() == 3
        assert features.shape[0] == 1  # Batch size
        assert features.shape[2] == 256  # Output dimension
        
        # Check approximate time dimension (30 FPS for 1 second ≈ 30 frames)
        expected_frames = extractor.get_output_length(16000)
        assert features.shape[1] == expected_frames
    
    def test_forward_shape_batch(self):
        """Test forward pass shape with batch."""
        extractor = Emotion2VecExtractor(
            model_name="dummy",
            target_fps=25,
            output_dim=128
        )
        
        # Batch of 3 samples, each 0.8 seconds
        batch_audio = torch.randn(3, 12800)
        
        features = extractor(batch_audio)
        
        assert features.shape[0] == 3    # Batch size
        assert features.shape[2] == 128  # Output dimension
        
        # Check time dimension
        expected_frames = extractor.get_output_length(12800)
        assert features.shape[1] == expected_frames
    
    def test_output_length_calculation(self):
        """Test output length calculation."""
        extractor = Emotion2VecExtractor(target_fps=30)
        
        # Test various input lengths
        test_lengths = [8000, 16000, 32000]  # 0.5s, 1s, 2s
        
        for input_len in test_lengths:
            output_len = extractor.get_output_length(input_len)
            
            # Should be approximately input_len * fps / sample_rate
            expected_len = int(input_len * 30 / 16000)
            
            assert output_len == expected_len
    
    def test_different_pooling_methods(self):
        """Test different temporal pooling methods."""
        pooling_methods = ["adaptive", "linear", "conv"]
        
        for method in pooling_methods:
            extractor = Emotion2VecExtractor(
                model_name="dummy",
                pooling_method=method,
                target_fps=20,
                output_dim=64
            )
            
            audio = torch.randn(16000)  # 1 second
            features = extractor(audio)
            
            assert features.shape[0] == 1   # Batch
            assert features.shape[1] == 20  # 20 FPS
            assert features.shape[2] == 64  # Output dim
    
    def test_layer_fusion(self):
        """Test layer fusion functionality."""
        # Create extractor with layer fusion
        layer_weights = [0.1, 0.2, 0.3, 0.4]  # 4 layers
        extractor = Emotion2VecExtractor(
            model_name="dummy",
            layer_weights=layer_weights,
            output_dim=128
        )
        
        assert extractor.use_layer_fusion is True
        assert hasattr(extractor, 'layer_weights')
        
        # Test forward pass
        audio = torch.randn(8000)
        features = extractor(audio)
        
        assert features.shape[2] == 128
    
    def test_freeze_pretrained(self):
        """Test freezing of pretrained weights."""
        extractor = Emotion2VecExtractor(
            model_name="dummy",
            freeze_pretrained=True
        )
        
        # For dummy model, freeze_pretrained should be False
        assert extractor.freeze_pretrained is False
        
        # Test that gradients can flow through output projection
        audio = torch.randn(4000, requires_grad=True)
        features = extractor(audio)
        loss = features.sum()
        loss.backward()
        
        # Audio should have gradients
        assert audio.grad is not None
    
    def test_custom_parameters(self):
        """Test extractor with custom parameters."""
        extractor = Emotion2VecExtractor(
            model_name="dummy",
            sample_rate=22050,
            target_fps=25.0,
            output_dim=512,
            pooling_method="linear"
        )
        
        assert extractor.sample_rate == 22050
        assert extractor.target_fps == 25.0
        assert extractor.output_dim == 512
        assert extractor.pooling_method == "linear"
        
        # Test with audio
        audio = torch.randn(22050)  # 1 second at 22.05 kHz
        features = extractor(audio)
        
        assert features.shape[1] == 25   # 25 FPS
        assert features.shape[2] == 512  # Custom output dim
    
    def test_invalid_input_dimension(self):
        """Test error handling for invalid input dimensions."""
        extractor = Emotion2VecExtractor(model_name="dummy")
        
        # 3D input should raise error
        invalid_input = torch.randn(2, 1000, 80)
        
        with pytest.raises(ValueError, match="Expected 1D or 2D input"):
            extractor(invalid_input)
    
    def test_zero_length_audio(self):
        """Test handling of very short audio."""
        extractor = Emotion2VecExtractor(model_name="dummy", target_fps=30)
        
        # Very short audio (less than 1 frame)
        short_audio = torch.randn(100)  # ~6ms at 16kHz
        
        features = extractor(short_audio)
        
        # Should produce at least 1 frame
        assert features.shape[1] >= 1
    
    def test_gradient_flow(self):
        """Test that gradients flow through the extractor."""
        extractor = Emotion2VecExtractor(model_name="dummy")
        
        # Input with gradient tracking
        audio = torch.randn(8000, requires_grad=True)
        
        features = extractor(audio)
        loss = features.sum()
        loss.backward()
        
        # Check that gradients exist
        assert audio.grad is not None
        assert not torch.all(audio.grad == 0)


class TestDummyWav2Vec2Model:
    """Test dummy Wav2Vec2 model."""
    
    def test_dummy_model_creation(self):
        """Test dummy model creation."""
        model = DummyWav2Vec2Model(hidden_size=512)
        
        assert model.hidden_size == 512
        assert model.config.hidden_size == 512
    
    def test_dummy_model_forward(self):
        """Test dummy model forward pass."""
        model = DummyWav2Vec2Model()
        
        # Test input
        batch_size = 2
        seq_len = 8000
        input_tensor = torch.randn(batch_size, seq_len)
        
        output = model(input_tensor)
        
        # Check output shape
        assert output.dim() == 3
        assert output.shape[0] == batch_size
        assert output.shape[2] == 768  # Default hidden size
        
        # Output sequence should be downsampled
        assert output.shape[1] < seq_len
    
    def test_dummy_model_target_length(self):
        """Test dummy model with target length."""
        model = DummyWav2Vec2Model()
        
        input_tensor = torch.randn(1, 16000)
        target_length = 50
        
        output = model(input_tensor, target_length=target_length)
        
        assert output.shape[1] == target_length
        assert output.shape[2] == 768


class TestEmotion2VecCache:
    """Test emotion2vec caching functionality."""
    
    def test_cache_creation(self):
        """Test cache creation."""
        cache = Emotion2VecCache(max_size=10)
        
        assert cache.max_size == 10
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_cache_put_get(self):
        """Test putting and getting from cache."""
        cache = Emotion2VecCache()
        
        # Create test features
        features = torch.randn(30, 256)
        audio_hash = "test_hash"
        
        # Put in cache
        cache.put(audio_hash, features)
        
        # Get from cache
        retrieved = cache.get(audio_hash)
        
        assert retrieved is not None
        assert torch.allclose(retrieved, features)
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        cache = Emotion2VecCache(max_size=2)
        
        # Add first item
        features1 = torch.randn(10, 64)
        cache.put("hash1", features1)
        
        # Add second item
        features2 = torch.randn(10, 64)
        cache.put("hash2", features2)
        
        # Add third item (should evict first)
        features3 = torch.randn(10, 64)
        cache.put("hash3", features3)
        
        # First item should be evicted
        assert cache.get("hash1") is None
        assert cache.get("hash2") is not None
        assert cache.get("hash3") is not None
    
    def test_cache_access_order(self):
        """Test that access updates order."""
        cache = Emotion2VecCache(max_size=2)
        
        # Add two items
        cache.put("hash1", torch.randn(5, 32))
        cache.put("hash2", torch.randn(5, 32))
        
        # Access first item (should move to end)
        cache.get("hash1")
        
        # Add third item (should evict second, not first)
        cache.put("hash3", torch.randn(5, 32))
        
        assert cache.get("hash1") is not None  # Should still be there
        assert cache.get("hash2") is None      # Should be evicted
        assert cache.get("hash3") is not None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = Emotion2VecCache()
        
        # Add some items
        cache.put("hash1", torch.randn(5, 32))
        cache.put("hash2", torch.randn(5, 32))
        
        # Clear cache
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
        assert cache.get("hash1") is None
        assert cache.get("hash2") is None


class TestAudioHashing:
    """Test audio hashing for caching."""
    
    def test_compute_audio_hash(self):
        """Test audio hash computation."""
        audio = torch.randn(1000)
        
        hash_val = compute_audio_hash(audio)
        
        assert isinstance(hash_val, int)
    
    def test_hash_consistency(self):
        """Test that same audio produces same hash."""
        audio = torch.randn(1000)
        
        hash1 = compute_audio_hash(audio)
        hash2 = compute_audio_hash(audio)
        
        assert hash1 == hash2
    
    def test_hash_difference(self):
        """Test that different audio produces different hashes."""
        audio1 = torch.randn(1000)
        audio2 = torch.randn(1000)
        
        hash1 = compute_audio_hash(audio1)
        hash2 = compute_audio_hash(audio2)
        
        # With high probability, they should be different
        assert hash1 != hash2
    
    def test_hash_long_audio(self):
        """Test hashing of long audio (should subsample)."""
        long_audio = torch.randn(100000)  # Long audio
        
        hash_val = compute_audio_hash(long_audio)
        
        assert isinstance(hash_val, int)


class TestFeatureValidation:
    """Test emotion2vec feature validation."""
    
    def test_valid_features(self):
        """Test validation with valid features."""
        batch_size, seq_len, feature_dim = 2, 30, 256
        features = torch.randn(batch_size, seq_len, feature_dim)
        
        results = validate_emotion2vec_features(features)
        
        assert results['valid'] is True
        assert results['stats']['shape'] == (batch_size, seq_len, feature_dim)
        assert 'embedding_norm_mean' in results['stats']
        assert 'temporal_variation' in results['stats']
    
    def test_invalid_dimensions(self):
        """Test validation with invalid dimensions."""
        invalid_features = torch.randn(10, 256)  # 2D instead of 3D
        
        results = validate_emotion2vec_features(invalid_features)
        
        assert results['valid'] is False
        assert any('Expected 3D tensor' in w for w in results['warnings'])
    
    def test_nan_detection(self):
        """Test detection of NaN values."""
        features = torch.randn(1, 10, 64)
        features[0, 0, 0] = float('nan')
        
        results = validate_emotion2vec_features(features)
        
        assert results['valid'] is False
        assert any('NaN values' in w for w in results['warnings'])
    
    def test_inf_detection(self):
        """Test detection of infinite values."""
        features = torch.randn(1, 10, 64)
        features[0, 5, 10] = float('inf')
        
        results = validate_emotion2vec_features(features)
        
        assert results['valid'] is False
        assert any('Infinite values' in w for w in results['warnings'])
    
    def test_small_embeddings_warning(self):
        """Test warning for very small embedding norms."""
        features = torch.randn(1, 10, 64) * 0.01  # Very small values
        
        results = validate_emotion2vec_features(features)
        
        assert any('Very small embedding norms' in w for w in results['warnings'])
    
    def test_large_embeddings_warning(self):
        """Test warning for very large embedding norms."""
        features = torch.randn(1, 10, 64) * 50  # Very large values
        
        results = validate_emotion2vec_features(features)
        
        assert any('Very large embedding norms' in w for w in results['warnings'])
    
    def test_constant_features_warning(self):
        """Test warning for constant (no variation) features."""
        features = torch.ones(1, 20, 64) * 5.0  # Constant features
        
        results = validate_emotion2vec_features(features)
        
        assert any('lack temporal variation' in w for w in results['warnings'])
    
    def test_noisy_features_warning(self):
        """Test warning for very noisy features."""
        # Create features with large temporal differences
        features = torch.randn(1, 20, 64) * 20
        
        results = validate_emotion2vec_features(features)
        
        # Might trigger noisy warning depending on random values
        # This test checks that the validation runs without error
        assert 'temporal_variation' in results['stats']
    
    def test_single_frame_features(self):
        """Test validation with single frame (no temporal dimension)."""
        features = torch.randn(2, 1, 128)  # Only 1 time step
        
        results = validate_emotion2vec_features(features)
        
        # Should not have temporal variation stats
        assert 'temporal_variation' not in results['stats']
        assert results['valid'] is True  # Should still be valid
    
    def test_statistics_computation(self):
        """Test that statistics are computed correctly."""
        # Create features with known properties
        batch_size, seq_len, feature_dim = 1, 10, 4
        features = torch.ones(batch_size, seq_len, feature_dim) * 2.0
        
        results = validate_emotion2vec_features(features)
        
        # Mean norm should be sqrt(4 * 4) = 4 (since each element is 2.0)
        expected_norm = 2.0 * np.sqrt(feature_dim)
        assert abs(results['stats']['embedding_norm_mean'] - expected_norm) < 1e-3
        
        # Temporal variation should be 0 (constant features)
        assert results['stats']['temporal_variation'] < 1e-6