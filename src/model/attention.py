"""
Cross-attention module for ARKit blendshape generation.

Unit name     : CrossAttentionBlendshape
Input         : Q(52,dq), K/V(T,dk) - queries are ARKit blendshapes, keys/values are audio
Output        : ΔBS(52,) - blendshape deltas
Dependencies  : torch.nn
Assumptions   : Causal attention, windowed for real-time
Failure modes : Attention collapse, gradient vanishing, memory issues
Test cases    : test_output_dim, test_masking, test_causal_constraint
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module optimized for blendshape generation.

    Uses ARKit blendshape parameters as queries and multi-stream audio features
    as keys/values. Supports causal masking and windowing for real-time inference.
    """

    def __init__(
        self,
        d_query: int = 128,
        d_key: int = 256,
        d_value: int = 256,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
        window_size: Optional[int] = None,
        temperature: float = 1.0,
        qkv_bias: bool = True,
    ):
        """
        Initialize multi-head cross-attention.

        Args:
            d_query: Dimension of query vectors (blendshape space)
            d_key: Dimension of key vectors (audio feature space)
            d_value: Dimension of value vectors (audio feature space)
            d_model: Model dimension for output projection
            num_heads: Number of attention heads
            dropout: Dropout probability
            causal: Whether to apply causal masking
            window_size: Size of attention window (None for full attention)
            temperature: Temperature scaling for attention weights
            qkv_bias: Whether to use bias in QKV projections
        """
        super().__init__()

        self.d_query = d_query
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size
        self.temperature = temperature

        # Check that model dimension is divisible by number of heads
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.head_dim = d_model // num_heads
        self.scale = (self.head_dim * temperature) ** -0.5

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(d_query, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_key, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_value, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights using Xavier/Glorot initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of cross-attention.

        Args:
            query: Query tensor of shape (B, Q, d_query) - blendshape parameters
            key: Key tensor of shape (B, T, d_key) - audio features
            value: Value tensor of shape (B, T, d_value) - audio features
            key_padding_mask: Mask for padded keys of shape (B, T)
            attn_mask: Additional attention mask of shape (Q, T) or (B, Q, T)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            - output: Output tensor of shape (B, Q, d_model)
            - attention_weights: Attention weights of shape (B, num_heads, Q, T) if requested
        """
        batch_size, q_len, _ = query.shape
        batch_size_k, seq_len, _ = key.shape

        if batch_size != batch_size_k:
            raise ValueError(
                f"Batch size mismatch: query {batch_size}, key {batch_size_k}"
            )

        # Project to query, key, value
        Q = self.q_proj(query)  # (B, Q, d_model)
        K = self.k_proj(key)  # (B, T, d_model)
        V = self.v_proj(value)  # (B, T, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, Q, d_head)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, T, d_head)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, H, T, d_head)

        # Compute attention
        output, attention_weights = self._compute_attention(
            Q, K, V, key_padding_mask, attn_mask, seq_len
        )

        # Reshape output
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        )  # (B, Q, d_model)

        # Final output projection
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        if return_attention:
            return output, attention_weights
        else:
            return output, None

    def _compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention with optional masking."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Q, T)

        # Apply causal mask if enabled
        if self.causal:
            causal_mask = self._create_causal_mask(Q.size(-2), seq_len, Q.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply window mask if specified
        if self.window_size is not None:
            window_mask = self._create_window_mask(
                Q.size(-2), seq_len, self.window_size, Q.device
            )
            scores = scores.masked_fill(window_mask, float("-inf"))

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) where True means valid, False means padded
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Apply additional attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Q, T)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, Q, T)
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, Q, T)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (B, H, Q, d_head)

        return output, attn_weights

    def _create_causal_mask(
        self, q_len: int, k_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens."""
        # For cross-attention, we typically don't need causal masking on queries
        # But we can mask future audio frames
        mask = torch.triu(torch.ones(q_len, k_len, device=device), diagonal=1).bool()
        return mask

    def _create_window_mask(
        self, q_len: int, k_len: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create windowed attention mask for local attention."""
        mask = torch.ones(q_len, k_len, device=device, dtype=torch.bool)

        # Allow attention only within window_size frames
        for i in range(q_len):
            # Compute corresponding time position in key sequence
            # Assuming queries are time-aligned with keys
            key_pos = int(i * k_len / q_len) if q_len > 0 else 0

            start = max(0, key_pos - window_size // 2)
            end = min(k_len, key_pos + window_size // 2 + 1)

            mask[i, start:end] = False  # False means allowed

        return mask


class MultiStreamAudioEncoder(nn.Module):
    """
    Audio encoder that processes multiple feature streams and fuses them.

    Handles mel-spectrogram, prosody, and emotion2vec features, applying
    appropriate preprocessing and fusion for cross-attention.
    """

    def __init__(
        self,
        mel_dim: int = 80,
        prosody_dim: int = 4,
        emotion_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_method: str = "concat",  # concat, add, gate
        use_positional_encoding: bool = True,
    ):
        """
        Initialize multi-stream audio encoder.

        Args:
            mel_dim: Dimension of mel-spectrogram features
            prosody_dim: Dimension of prosody features
            emotion_dim: Dimension of emotion2vec features
            d_model: Model dimension
            num_layers: Number of encoder layers
            dropout: Dropout probability
            fusion_method: Method to fuse multi-stream features
            use_positional_encoding: Whether to add positional encoding
        """
        super().__init__()

        self.mel_dim = mel_dim
        self.prosody_dim = prosody_dim
        self.emotion_dim = emotion_dim
        self.d_model = d_model
        self.fusion_method = fusion_method
        self.use_positional_encoding = use_positional_encoding

        # Individual stream encoders
        self.mel_encoder = nn.Sequential(
            nn.Linear(mel_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )

        self.prosody_encoder = nn.Sequential(
            nn.Linear(prosody_dim, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model),
        )

        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
        )

        # Fusion layer
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(d_model * 3, d_model)
        elif fusion_method == "gate":
            self.gate_mel = nn.Linear(d_model, 1)
            self.gate_prosody = nn.Linear(d_model, 1)
            self.gate_emotion = nn.Linear(d_model, 1)
        # "add" method doesn't need additional parameters

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        mel_features: torch.Tensor,
        prosody_features: torch.Tensor,
        emotion_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-stream encoder.

        Args:
            mel_features: Mel-spectrogram features of shape (B, T, mel_dim)
            prosody_features: Prosody features of shape (B, T, prosody_dim)
            emotion_features: Emotion features of shape (B, T, emotion_dim)
            mask: Padding mask of shape (B, T)

        Returns:
            Encoded features of shape (B, T, d_model)
        """
        batch_size, seq_len = mel_features.shape[:2]

        # Encode individual streams
        mel_encoded = self.mel_encoder(mel_features)  # (B, T, d_model)
        prosody_encoded = self.prosody_encoder(prosody_features)  # (B, T, d_model)
        emotion_encoded = self.emotion_encoder(emotion_features)  # (B, T, d_model)

        # Fuse streams
        if self.fusion_method == "concat":
            # Concatenate and project
            fused = torch.cat([mel_encoded, prosody_encoded, emotion_encoded], dim=-1)
            fused = self.fusion_proj(fused)
        elif self.fusion_method == "add":
            # Simple addition
            fused = mel_encoded + prosody_encoded + emotion_encoded
        elif self.fusion_method == "gate":
            # Gated fusion
            gate_mel = torch.sigmoid(self.gate_mel(mel_encoded))
            gate_prosody = torch.sigmoid(self.gate_prosody(prosody_encoded))
            gate_emotion = torch.sigmoid(self.gate_emotion(emotion_encoded))

            # Normalize gates
            gate_sum = gate_mel + gate_prosody + gate_emotion + 1e-8
            gate_mel = gate_mel / gate_sum
            gate_prosody = gate_prosody / gate_sum
            gate_emotion = gate_emotion / gate_sum

            fused = (
                gate_mel * mel_encoded
                + gate_prosody * prosody_encoded
                + gate_emotion * emotion_encoded
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Add positional encoding
        if self.use_positional_encoding:
            fused = self.pos_encoder(fused)

        # Apply transformer encoder
        if mask is not None:
            # Convert mask to attention mask (True means ignore)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer(fused, src_key_padding_mask=src_key_padding_mask)

        # Layer normalization
        encoded = self.layer_norm(encoded)

        return encoded


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class BlendshapeQueryEmbedding(nn.Module):
    """
    Learnable embedding for ARKit blendshape queries.

    Provides learnable query vectors for each of the 52 ARKit blendshapes,
    optionally conditioned on previous blendshape state.
    """

    def __init__(
        self,
        num_blendshapes: int = 52,
        d_query: int = 128,
        use_conditioning: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize blendshape query embedding.

        Args:
            num_blendshapes: Number of blendshape parameters (52 for ARKit)
            d_query: Dimension of query vectors
            use_conditioning: Whether to condition on previous blendshape state
            dropout: Dropout probability
        """
        super().__init__()

        self.num_blendshapes = num_blendshapes
        self.d_query = d_query
        self.use_conditioning = use_conditioning

        # Learnable query embeddings for each blendshape
        self.query_embeddings = nn.Parameter(torch.randn(num_blendshapes, d_query))

        # Optional conditioning network
        if use_conditioning:
            self.conditioning_net = nn.Sequential(
                nn.Linear(num_blendshapes, d_query // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_query // 2, d_query),
            )

        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.query_embeddings)

    def forward(
        self,
        batch_size: int,
        prev_blendshapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate query embeddings for blendshapes.

        Args:
            batch_size: Batch size
            prev_blendshapes: Previous blendshape state of shape (B, 52)

        Returns:
            Query embeddings of shape (B, 52, d_query)
        """
        device = self.query_embeddings.device

        # Start with base query embeddings
        queries = self.query_embeddings.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, 52, d_query)

        # Add conditioning if available
        if self.use_conditioning and prev_blendshapes is not None:
            conditioning = self.conditioning_net(prev_blendshapes)  # (B, d_query)
            conditioning = conditioning.unsqueeze(1).repeat(
                1, self.num_blendshapes, 1
            )  # (B, 52, d_query)
            queries = queries + conditioning

        queries = self.dropout(queries)

        return queries


def create_attention_mask(
    seq_length: int,
    window_size: Optional[int] = None,
    causal: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create attention mask for transformer models.

    Args:
        seq_length: Sequence length
        window_size: Window size for local attention (None for full attention)
        causal: Whether to apply causal masking
        device: Device to create mask on

    Returns:
        Attention mask of shape (seq_length, seq_length)
    """
    mask = torch.zeros(seq_length, seq_length, device=device, dtype=torch.bool)

    if causal:
        # Upper triangular mask (prevent attending to future)
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device), diagonal=1
        ).bool()

    if window_size is not None:
        # Create local attention window
        for i in range(seq_length):
            start = max(0, i - window_size // 2)
            end = min(seq_length, i + window_size // 2 + 1)

            # Allow attention within window
            window_mask = torch.ones(seq_length, device=device, dtype=torch.bool)
            window_mask[start:end] = False
            mask[i] = mask[i] | window_mask

    return mask
