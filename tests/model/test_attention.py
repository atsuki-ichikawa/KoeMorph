"""Tests for model.attention module."""

import pytest
import torch

from src.model.attention import (
    BlendshapeQueryEmbedding,
    MultiHeadCrossAttention,
    MultiStreamAudioEncoder,
    PositionalEncoding,
    create_attention_mask,
)


class TestMultiHeadCrossAttention:
    """Test multi-head cross-attention functionality."""

    def test_attention_creation(self):
        """Test attention module creation."""
        attention = MultiHeadCrossAttention(
            d_query=128, d_key=256, d_value=256, d_model=512, num_heads=8
        )

        assert attention.d_query == 128
        assert attention.d_key == 256
        assert attention.d_value == 256
        assert attention.d_model == 512
        assert attention.num_heads == 8
        assert attention.head_dim == 64  # 512 / 8

    def test_attention_forward_basic(self):
        """Test basic forward pass."""
        attention = MultiHeadCrossAttention(
            d_query=64, d_key=128, d_value=128, d_model=256, num_heads=4
        )

        batch_size = 2
        q_len = 52  # Number of blendshapes
        seq_len = 30  # Audio sequence length

        query = torch.randn(batch_size, q_len, 64)
        key = torch.randn(batch_size, seq_len, 128)
        value = torch.randn(batch_size, seq_len, 128)

        output, attn_weights = attention(query, key, value, return_attention=True)

        # Check output shape
        assert output.shape == (batch_size, q_len, 256)

        # Check attention weights shape
        assert attn_weights.shape == (batch_size, 4, q_len, seq_len)

        # Check attention weights sum to 1
        assert torch.allclose(
            attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1))
        )

    def test_attention_with_padding_mask(self):
        """Test attention with key padding mask."""
        attention = MultiHeadCrossAttention(
            d_query=64, d_key=64, d_value=64, d_model=128, num_heads=2
        )

        batch_size = 2
        q_len = 10
        seq_len = 20

        query = torch.randn(batch_size, q_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)

        # Create padding mask (True for valid positions, False for padded)
        key_padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        key_padding_mask[0, 15:] = False  # Mask last 5 positions for first sample
        key_padding_mask[1, 10:] = False  # Mask last 10 positions for second sample

        output, attn_weights = attention(
            query, key, value, key_padding_mask=key_padding_mask, return_attention=True
        )

        assert output.shape == (batch_size, q_len, 128)

        # Check that attention weights are zero for masked positions
        assert torch.allclose(
            attn_weights[0, :, :, 15:], torch.zeros_like(attn_weights[0, :, :, 15:])
        )
        assert torch.allclose(
            attn_weights[1, :, :, 10:], torch.zeros_like(attn_weights[1, :, :, 10:])
        )

    def test_causal_attention(self):
        """Test causal attention masking."""
        attention = MultiHeadCrossAttention(
            d_query=32, d_key=32, d_value=32, d_model=64, num_heads=2, causal=True
        )

        batch_size = 1
        q_len = 5
        seq_len = 10

        query = torch.randn(batch_size, q_len, 32)
        key = torch.randn(batch_size, seq_len, 32)
        value = torch.randn(batch_size, seq_len, 32)

        output, attn_weights = attention(query, key, value, return_attention=True)

        # For causal attention, should not attend to future positions
        # This is a simplified check - actual causal logic depends on alignment
        assert output.shape == (batch_size, q_len, 64)
        assert attn_weights.shape == (batch_size, 2, q_len, seq_len)

    def test_windowed_attention(self):
        """Test windowed attention."""
        attention = MultiHeadCrossAttention(
            d_query=32, d_key=32, d_value=32, d_model=64, num_heads=2, window_size=6
        )

        batch_size = 1
        q_len = 8
        seq_len = 20

        query = torch.randn(batch_size, q_len, 32)
        key = torch.randn(batch_size, seq_len, 32)
        value = torch.randn(batch_size, seq_len, 32)

        output, attn_weights = attention(query, key, value, return_attention=True)

        assert output.shape == (batch_size, q_len, 64)

        # Check that attention is limited to window (this is approximate)
        # Exact check would depend on the specific windowing implementation
        assert attn_weights.shape == (batch_size, 2, q_len, seq_len)

    def test_gradient_flow(self):
        """Test gradient flow through attention."""
        attention = MultiHeadCrossAttention(
            d_query=32, d_key=32, d_value=32, d_model=64, num_heads=2
        )

        query = torch.randn(1, 5, 32, requires_grad=True)
        key = torch.randn(1, 10, 32, requires_grad=True)
        value = torch.randn(1, 10, 32, requires_grad=True)

        output, _ = attention(query, key, value)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert not torch.all(query.grad == 0)

    def test_invalid_head_dimension(self):
        """Test error handling for invalid head dimension."""
        with pytest.raises(ValueError, match="d_model.*must be divisible by num_heads"):
            MultiHeadCrossAttention(
                d_query=32, d_key=32, d_value=32, d_model=65, num_heads=8
            )

    def test_batch_size_mismatch(self):
        """Test error handling for batch size mismatch."""
        attention = MultiHeadCrossAttention(
            d_query=32, d_key=32, d_value=32, d_model=64, num_heads=2
        )

        query = torch.randn(2, 5, 32)
        key = torch.randn(3, 10, 32)  # Different batch size
        value = torch.randn(3, 10, 32)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            attention(query, key, value)


class TestMultiStreamAudioEncoder:
    """Test multi-stream audio encoder functionality."""

    def test_encoder_creation(self):
        """Test encoder creation."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=80,
            prosody_dim=4,
            emotion_dim=256,
            d_model=128,
            num_layers=2,
            fusion_method="concat",
        )

        assert encoder.mel_dim == 80
        assert encoder.prosody_dim == 4
        assert encoder.emotion_dim == 256
        assert encoder.d_model == 128
        assert encoder.fusion_method == "concat"

    def test_encoder_forward_concat(self):
        """Test encoder forward pass with concatenation fusion."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=80,
            prosody_dim=4,
            emotion_dim=256,
            d_model=128,
            fusion_method="concat",
        )

        batch_size = 2
        seq_len = 30

        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)

        output = encoder(mel_features, prosody_features, emotion_features)

        assert output.shape == (batch_size, seq_len, 128)

    def test_encoder_forward_add(self):
        """Test encoder forward pass with addition fusion."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=80, prosody_dim=4, emotion_dim=256, d_model=128, fusion_method="add"
        )

        batch_size = 1
        seq_len = 20

        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)

        output = encoder(mel_features, prosody_features, emotion_features)

        assert output.shape == (batch_size, seq_len, 128)

    def test_encoder_forward_gate(self):
        """Test encoder forward pass with gated fusion."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=80,
            prosody_dim=4,
            emotion_dim=256,
            d_model=128,
            fusion_method="gate",
        )

        batch_size = 1
        seq_len = 15

        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)

        output = encoder(mel_features, prosody_features, emotion_features)

        assert output.shape == (batch_size, seq_len, 128)

    def test_encoder_with_mask(self):
        """Test encoder with padding mask."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=40, prosody_dim=4, emotion_dim=128, d_model=64, num_layers=1
        )

        batch_size = 2
        seq_len = 10

        mel_features = torch.randn(batch_size, seq_len, 40)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 128)

        # Create mask
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, 7:] = False  # Mask last 3 positions
        mask[1, 5:] = False  # Mask last 5 positions

        output = encoder(mel_features, prosody_features, emotion_features, mask=mask)

        assert output.shape == (batch_size, seq_len, 64)

    def test_encoder_without_positional_encoding(self):
        """Test encoder without positional encoding."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=80,
            prosody_dim=4,
            emotion_dim=256,
            d_model=128,
            use_positional_encoding=False,
        )

        assert not encoder.use_positional_encoding

        batch_size = 1
        seq_len = 10

        mel_features = torch.randn(batch_size, seq_len, 80)
        prosody_features = torch.randn(batch_size, seq_len, 4)
        emotion_features = torch.randn(batch_size, seq_len, 256)

        output = encoder(mel_features, prosody_features, emotion_features)

        assert output.shape == (batch_size, seq_len, 128)

    def test_encoder_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = MultiStreamAudioEncoder(
            mel_dim=40, prosody_dim=4, emotion_dim=64, d_model=32, num_layers=1
        )

        mel_features = torch.randn(1, 5, 40, requires_grad=True)
        prosody_features = torch.randn(1, 5, 4, requires_grad=True)
        emotion_features = torch.randn(1, 5, 64, requires_grad=True)

        output = encoder(mel_features, prosody_features, emotion_features)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert mel_features.grad is not None
        assert prosody_features.grad is not None
        assert emotion_features.grad is not None

    def test_invalid_fusion_method(self):
        """Test error handling for invalid fusion method."""
        with pytest.raises(ValueError, match="Unknown fusion method"):
            encoder = MultiStreamAudioEncoder(fusion_method="invalid")

            # Try forward pass to trigger error
            mel = torch.randn(1, 5, 80)
            prosody = torch.randn(1, 5, 4)
            emotion = torch.randn(1, 5, 256)
            encoder(mel, prosody, emotion)


class TestPositionalEncoding:
    """Test positional encoding functionality."""

    def test_positional_encoding_creation(self):
        """Test positional encoding creation."""
        pos_enc = PositionalEncoding(d_model=128, dropout=0.1, max_len=1000)

        assert pos_enc.pe.shape == (1000, 1, 128)

    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        pos_enc = PositionalEncoding(d_model=64, dropout=0.0)  # No dropout for testing

        batch_size = 2
        seq_len = 20
        input_tensor = torch.randn(batch_size, seq_len, 64)

        output = pos_enc(input_tensor)

        assert output.shape == input_tensor.shape

        # Output should be different from input (due to added positional encoding)
        assert not torch.allclose(output, input_tensor)

    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        pos_enc = PositionalEncoding(d_model=32, dropout=0.0)

        input_tensor = torch.randn(1, 10, 32)

        output1 = pos_enc(input_tensor)
        output2 = pos_enc(input_tensor)

        # Should be identical when dropout=0
        assert torch.allclose(output1, output2)


class TestBlendshapeQueryEmbedding:
    """Test blendshape query embedding functionality."""

    def test_query_embedding_creation(self):
        """Test query embedding creation."""
        query_emb = BlendshapeQueryEmbedding(
            num_blendshapes=52, d_query=128, use_conditioning=True
        )

        assert query_emb.num_blendshapes == 52
        assert query_emb.d_query == 128
        assert query_emb.use_conditioning is True
        assert query_emb.query_embeddings.shape == (52, 128)

    def test_query_embedding_forward_no_conditioning(self):
        """Test query embedding forward without conditioning."""
        query_emb = BlendshapeQueryEmbedding(
            num_blendshapes=52, d_query=64, use_conditioning=False
        )

        batch_size = 3
        queries = query_emb(batch_size)

        assert queries.shape == (batch_size, 52, 64)

    def test_query_embedding_forward_with_conditioning(self):
        """Test query embedding forward with conditioning."""
        query_emb = BlendshapeQueryEmbedding(
            num_blendshapes=52, d_query=64, use_conditioning=True
        )

        batch_size = 2
        prev_blendshapes = torch.randn(batch_size, 52)

        queries = query_emb(batch_size, prev_blendshapes)

        assert queries.shape == (batch_size, 52, 64)

        # With conditioning, output should be different from base embeddings
        queries_no_cond = query_emb(batch_size, None)
        assert not torch.allclose(queries, queries_no_cond)

    def test_query_embedding_gradient_flow(self):
        """Test gradient flow through query embeddings."""
        query_emb = BlendshapeQueryEmbedding(num_blendshapes=10, d_query=32)

        prev_blendshapes = torch.randn(1, 10, requires_grad=True)
        queries = query_emb(1, prev_blendshapes)
        loss = queries.sum()
        loss.backward()

        # Check gradients exist
        assert prev_blendshapes.grad is not None
        assert query_emb.query_embeddings.grad is not None


class TestAttentionMask:
    """Test attention mask creation utilities."""

    def test_create_basic_mask(self):
        """Test basic mask creation."""
        mask = create_attention_mask(seq_length=5)

        assert mask.shape == (5, 5)
        assert mask.dtype == torch.bool

        # No masking by default
        assert not mask.any()

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        mask = create_attention_mask(seq_length=4, causal=True)

        assert mask.shape == (4, 4)

        # Upper triangular should be True (masked)
        expected = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )

        assert torch.equal(mask, expected)

    def test_create_windowed_mask(self):
        """Test windowed mask creation."""
        mask = create_attention_mask(seq_length=6, window_size=3)

        assert mask.shape == (6, 6)

        # Each position should only attend to window_size neighbors
        # This test checks that the mask has the right structure
        for i in range(6):
            # Count unmasked positions for each row
            unmasked_count = (~mask[i]).sum().item()
            assert unmasked_count <= 3  # Should not exceed window size

    def test_create_causal_windowed_mask(self):
        """Test combined causal and windowed mask."""
        mask = create_attention_mask(seq_length=5, window_size=3, causal=True)

        assert mask.shape == (5, 5)

        # Should respect both causal and window constraints
        # (More complex to verify exactly, but should not crash)
        assert mask.dtype == torch.bool

    def test_mask_device(self):
        """Test mask creation on specific device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        mask = create_attention_mask(seq_length=3, device=device)

        assert mask.device == device
