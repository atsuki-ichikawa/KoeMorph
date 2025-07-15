"""
Emotion2Vec feature extraction for emotional speech representation.

Unit name     : Emotion2VecExtractor
Input         : torch.FloatTensor (B, L)   [16 kHz audio]
Output        : torch.FloatTensor (B, T, 256) [emotion embeddings]
Dependencies  : transformers, fairseq (optional)
Assumptions   : Pre-trained emotion2vec model available
Failure modes : Model loading errors, CUDA OOM, long sequences
Test cases    : test_shape, test_embedding_norm, test_batch_consistency
"""

import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not available. Emotion2Vec will use dummy embeddings.")


class Emotion2VecExtractor(nn.Module):
    """
    Emotion2Vec feature extractor for emotional speech representation.

    Extracts high-level emotional and prosodic features from speech using
    pre-trained emotion2vec model, with temporal alignment to target FPS.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wav2vec2-base",  # Fallback model
        target_fps: float = 30.0,
        sample_rate: int = 16000,
        freeze_pretrained: bool = True,
        output_dim: int = 256,
        pooling_method: str = "adaptive",  # adaptive, linear, conv
        layer_weights: Optional[list] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Emotion2Vec extractor.

        Args:
            model_name: Pre-trained model name or path
            target_fps: Target output frame rate
            sample_rate: Audio sample rate (must match model)
            freeze_pretrained: Whether to freeze pre-trained weights
            output_dim: Output embedding dimension
            pooling_method: Method to downsample to target FPS
            layer_weights: Weights for layer fusion (None for last layer only)
            cache_dir: Directory to cache downloaded models
        """
        super().__init__()

        self.model_name = model_name
        self.target_fps = target_fps
        self.sample_rate = sample_rate
        self.freeze_pretrained = freeze_pretrained
        self.output_dim = output_dim
        self.pooling_method = pooling_method

        # Initialize processor and model
        if HAS_TRANSFORMERS:
            self._init_transformers_model(cache_dir)
        else:
            self._init_dummy_model()

        # Layer fusion weights
        if layer_weights is not None:
            self.register_parameter(
                "layer_weights",
                nn.Parameter(torch.tensor(layer_weights, dtype=torch.float32)),
            )
            self.use_layer_fusion = True
        else:
            self.use_layer_fusion = False

        # Temporal pooling layer
        self._init_pooling_layer()

        # Output projection
        hidden_dim = self.base_model.config.hidden_size if HAS_TRANSFORMERS else 768
        if output_dim != hidden_dim:
            self.output_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_projection = nn.Identity()

    def _init_transformers_model(self, cache_dir: Optional[str]):
        """Initialize model using transformers library."""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.model_name, cache_dir=cache_dir
            )
            self.base_model = Wav2Vec2Model.from_pretrained(
                self.model_name, cache_dir=cache_dir
            )

            # Freeze pretrained weights if requested
            if self.freeze_pretrained:
                for param in self.base_model.parameters():
                    param.requires_grad = False

            # Set model to eval mode for feature extraction
            self.base_model.eval()

        except Exception as e:
            warnings.warn(f"Failed to load {self.model_name}: {e}. Using dummy model.")
            self._init_dummy_model()

    def _init_dummy_model(self):
        """Initialize dummy model for testing/fallback."""
        self.processor = None
        self.base_model = DummyWav2Vec2Model()
        self.freeze_pretrained = False

    def _init_pooling_layer(self):
        """Initialize temporal pooling layer based on method."""
        if self.pooling_method == "adaptive":
            # Adaptive pooling to target sequence length
            self.temporal_pooling = nn.AdaptiveAvgPool1d(1)  # Will be set dynamically
        elif self.pooling_method == "linear":
            # Linear interpolation
            self.temporal_pooling = nn.Identity()  # Handle in forward
        elif self.pooling_method == "conv":
            # Convolutional downsampling
            self.temporal_pooling = nn.Conv1d(
                in_channels=768,  # Wav2Vec2 hidden size
                out_channels=768,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract emotion2vec features from waveform.

        Args:
            waveform: Input waveform of shape (B, L) or (L,)

        Returns:
            Emotion embeddings of shape (B, T, output_dim)
        """
        # Handle single sample input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2:
            raise ValueError(f"Expected 1D or 2D input, got {waveform.dim()}D")

        batch_size = waveform.shape[0]
        device = waveform.device

        # Move base model to same device
        if hasattr(self.base_model, "to"):
            self.base_model = self.base_model.to(device)

        # Extract features
        if HAS_TRANSFORMERS and self.processor is not None:
            features = self._extract_transformers_features(waveform)
        else:
            features = self._extract_dummy_features(waveform)

        # Apply temporal pooling to target FPS
        features_pooled = self._apply_temporal_pooling(features, waveform.shape[1])

        # Apply output projection
        features_projected = self.output_projection(features_pooled)

        return features_projected

    def _extract_transformers_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features using transformers Wav2Vec2."""
        batch_size = waveform.shape[0]
        features_list = []

        # Process each sample in batch (for compatibility with processor)
        for i in range(batch_size):
            # Convert to numpy for processor
            audio_np = waveform[i].detach().cpu().numpy()

            # Process audio
            inputs = self.processor(
                audio_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            input_values = inputs.input_values.to(waveform.device)

            # Extract features
            with torch.no_grad() if self.freeze_pretrained else torch.enable_grad():
                outputs = self.base_model(input_values, output_hidden_states=True)

                if self.use_layer_fusion:
                    # Weighted sum of all hidden states
                    hidden_states = outputs.hidden_states  # List of (1, T, D)
                    hidden_states = torch.stack(
                        hidden_states, dim=0
                    )  # (num_layers, 1, T, D)

                    # Apply layer weights
                    weights = F.softmax(self.layer_weights, dim=0)  # (num_layers,)
                    features = torch.sum(
                        weights.view(-1, 1, 1, 1) * hidden_states, dim=0
                    )  # (1, T, D)
                else:
                    # Use last hidden state
                    features = outputs.last_hidden_state  # (1, T, D)

            features_list.append(features.squeeze(0))  # (T, D)

        # Stack batch dimension
        # Pad sequences to same length
        max_len = max(f.shape[0] for f in features_list)
        features_padded = []

        for features in features_list:
            if features.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - features.shape[0],
                    features.shape[1],
                    device=features.device,
                    dtype=features.dtype,
                )
                features = torch.cat([features, padding], dim=0)
            features_padded.append(features)

        return torch.stack(features_padded, dim=0)  # (B, T, D)

    def _extract_dummy_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract dummy features for testing/fallback."""
        batch_size, seq_len = waveform.shape

        # Simulate Wav2Vec2-like downsampling (roughly 50 Hz)
        downsample_factor = self.sample_rate // 50  # ~320 for 16kHz
        feature_len = seq_len // downsample_factor

        # Generate dummy features with some correlation to input
        features = self.base_model(waveform, feature_len)

        return features

    def _apply_temporal_pooling(
        self, features: torch.Tensor, audio_length: int
    ) -> torch.Tensor:
        """Apply temporal pooling to match target FPS."""
        batch_size, current_seq_len, hidden_dim = features.shape

        # Calculate target sequence length
        audio_duration = audio_length / self.sample_rate
        target_seq_len = int(audio_duration * self.target_fps)

        if target_seq_len == 0:
            return torch.zeros(batch_size, 1, hidden_dim, device=features.device)

        if self.pooling_method == "adaptive":
            # Reshape for adaptive pooling: (B, D, T)
            features_transposed = features.transpose(1, 2)

            # Apply adaptive pooling
            pooled = F.adaptive_avg_pool1d(features_transposed, target_seq_len)

            # Reshape back: (B, T, D)
            return pooled.transpose(1, 2)

        elif self.pooling_method == "linear":
            # Linear interpolation
            if current_seq_len == target_seq_len:
                return features

            # Interpolate along time dimension
            features_transposed = features.transpose(1, 2)  # (B, D, T)
            interpolated = F.interpolate(
                features_transposed,
                size=target_seq_len,
                mode="linear",
                align_corners=False,
            )
            return interpolated.transpose(1, 2)  # (B, T, D)

        elif self.pooling_method == "conv":
            # Apply convolutional pooling
            features_transposed = features.transpose(1, 2)  # (B, D, T)
            pooled = self.temporal_pooling(features_transposed)

            # Further adjust to exact target length if needed
            if pooled.shape[2] != target_seq_len:
                pooled = F.adaptive_avg_pool1d(pooled, target_seq_len)

            return pooled.transpose(1, 2)  # (B, T, D)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length for given input length."""
        audio_duration = input_length / self.sample_rate
        return int(audio_duration * self.target_fps)


class DummyWav2Vec2Model(nn.Module):
    """Dummy Wav2Vec2-like model for testing."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # Simple CNN to simulate feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, hidden_size, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # Mock config for compatibility
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(
        self, waveform: torch.Tensor, target_length: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass of dummy model."""
        batch_size, seq_len = waveform.shape

        # Add channel dimension
        x = waveform.unsqueeze(1)  # (B, 1, L)

        # Apply conv layers
        x = self.conv_layers(x)  # (B, hidden_size, T')

        # Transpose to (B, T', hidden_size)
        x = x.transpose(1, 2)

        # Adjust length if specified
        if target_length is not None and x.shape[1] != target_length:
            x_transposed = x.transpose(1, 2)  # (B, hidden_size, T')
            x_resized = F.adaptive_avg_pool1d(x_transposed, target_length)
            x = x_resized.transpose(1, 2)  # (B, target_length, hidden_size)

        return x


class Emotion2VecCache:
    """
    Cache for emotion2vec features to avoid recomputation.

    Useful for training where same audio might be processed multiple times.
    """

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, audio_hash: str) -> Optional[torch.Tensor]:
        """Get cached features by audio hash."""
        if audio_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(audio_hash)
            self.access_order.append(audio_hash)
            return self.cache[audio_hash]
        return None

    def put(self, audio_hash: str, features: torch.Tensor):
        """Cache features with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[audio_hash] = features.clone()
        self.access_order.append(audio_hash)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


def compute_audio_hash(waveform: torch.Tensor) -> str:
    """Compute hash of audio waveform for caching."""
    # Use a subset of audio for efficiency
    if len(waveform) > 1000:
        sample_audio = waveform[:: len(waveform) // 1000]
    else:
        sample_audio = waveform

    # Compute hash of samples
    hash_input = (sample_audio * 1000).int()  # Quantize for consistent hashing
    return hash(tuple(hash_input.tolist()))


def validate_emotion2vec_features(features: torch.Tensor) -> dict:
    """
    Validate emotion2vec features for quality and consistency.

    Args:
        features: Emotion features of shape (B, T, D)

    Returns:
        Dictionary with validation results
    """
    results = {"valid": True, "warnings": [], "stats": {}}

    if features.dim() != 3:
        results["valid"] = False
        results["warnings"].append(f"Expected 3D tensor, got {features.dim()}D")
        return results

    batch_size, seq_len, feature_dim = features.shape
    results["stats"]["shape"] = (batch_size, seq_len, feature_dim)

    # Check for NaN or infinite values
    if torch.isnan(features).any():
        results["valid"] = False
        results["warnings"].append("NaN values detected")

    if torch.isinf(features).any():
        results["valid"] = False
        results["warnings"].append("Infinite values detected")

    # Compute feature statistics
    feature_mean = features.mean(dim=(0, 1))  # Mean across batch and time
    feature_std = features.std(dim=(0, 1))  # Std across batch and time

    results["stats"]["mean_norm"] = torch.norm(feature_mean).item()
    results["stats"]["std_mean"] = feature_std.mean().item()

    # Check feature magnitudes
    feature_norm = torch.norm(features, dim=-1)  # (B, T)
    mean_norm = feature_norm.mean().item()
    results["stats"]["embedding_norm_mean"] = mean_norm

    # Warn about unusual magnitudes
    if mean_norm < 0.1:
        results["warnings"].append(f"Very small embedding norms: {mean_norm:.3f}")
    elif mean_norm > 100:
        results["warnings"].append(f"Very large embedding norms: {mean_norm:.3f}")

    # Check temporal consistency
    if seq_len > 1:
        temporal_diff = torch.diff(features, dim=1)  # (B, T-1, D)
        temporal_variation = torch.norm(temporal_diff, dim=-1).mean().item()
        results["stats"]["temporal_variation"] = temporal_variation

        # Features should have some variation but not be too noisy
        if temporal_variation < 0.01:
            results["warnings"].append(
                "Features lack temporal variation (may be constant)"
            )
        elif temporal_variation > 10:
            results["warnings"].append("Features are very noisy temporally")

    return results
