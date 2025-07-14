"""
Blendshape decoder for converting attention output to final blendshape coefficients.

Unit name     : BlendshapeDecoder
Input         : torch.FloatTensor (B, 52, d_model) [attention output]
Output        : torch.FloatTensor (B, 52) [blendshape coefficients 0-1]
Dependencies  : torch.nn
Assumptions   : Input from cross-attention, output range [0,1]
Failure modes : Gradient vanishing, value clamping, smoothing artifacts
Test cases    : test_output_range, test_smoothing, test_residual_connection
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlendshapeDecoder(nn.Module):
    """
    Decoder that converts attention output to final blendshape coefficients.
    
    Applies MLP layers, residual connections, and temporal smoothing to produce
    stable blendshape outputs in the [0,1] range suitable for real-time rendering.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_blendshapes: int = 52,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        output_activation: str = "sigmoid",  # sigmoid, tanh, none
        use_residual: bool = True,
        use_layer_norm: bool = True,
        bias: bool = True,
    ):
        """
        Initialize blendshape decoder.
        
        Args:
            d_model: Input dimension from attention
            hidden_dim: Hidden layer dimension
            num_blendshapes: Number of output blendshapes (52 for ARKit)
            num_layers: Number of hidden layers
            activation: Activation function name
            dropout: Dropout probability
            output_activation: Final activation (sigmoid for [0,1], tanh for [-1,1])
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_blendshapes = num_blendshapes
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input projection
        self.input_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        for i in range(num_layers):
            self.hidden_layers.append(
                nn.Linear(hidden_dim, hidden_dim, bias=bias)
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_blendshapes, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        attention_output: torch.Tensor,
        prev_blendshapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode attention output to blendshape coefficients.
        
        Args:
            attention_output: Output from cross-attention of shape (B, 52, d_model)
            prev_blendshapes: Previous blendshape state of shape (B, 52) for residual
            
        Returns:
            Blendshape coefficients of shape (B, 52)
        """
        batch_size, seq_len, _ = attention_output.shape
        
        if seq_len != self.num_blendshapes:
            raise ValueError(f"Expected {self.num_blendshapes} blendshapes, got {seq_len}")
        
        # Input projection
        x = self.input_proj(attention_output)  # (B, 52, hidden_dim)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            residual = x
            
            x = layer(x)
            
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout(x)
            
            # Residual connection
            if self.use_residual:
                x = x + residual
        
        # Output projection (per-blendshape)
        x = self.output_proj(x)  # (B, 52, 52) - each blendshape gets its own prediction
        
        # Take diagonal to get per-blendshape predictions
        # This ensures each blendshape only depends on its own attention output
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        bs_indices = torch.arange(self.num_blendshapes, device=x.device).unsqueeze(0)
        
        blendshapes = x[batch_indices, bs_indices, bs_indices]  # (B, 52)
        
        # Apply output activation
        if self.output_activation == "sigmoid":
            blendshapes = torch.sigmoid(blendshapes)
        elif self.output_activation == "tanh":
            blendshapes = torch.tanh(blendshapes)
        elif self.output_activation == "none":
            pass  # No activation
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")
        
        # Optional residual connection with previous blendshapes
        if prev_blendshapes is not None and self.use_residual:
            # Weighted combination with previous state
            alpha = 0.1  # Small weight for stability
            blendshapes = (1 - alpha) * blendshapes + alpha * prev_blendshapes
        
        return blendshapes


class TemporalSmoother(nn.Module):
    """
    Temporal smoothing module for blendshape sequences.
    
    Applies various smoothing techniques to reduce jitter and ensure
    temporal consistency in blendshape animations.
    """
    
    def __init__(
        self,
        num_blendshapes: int = 52,
        smoothing_method: str = "exponential",  # exponential, gaussian, median
        alpha: float = 0.8,
        window_size: int = 5,
        learnable: bool = False,
    ):
        """
        Initialize temporal smoother.
        
        Args:
            num_blendshapes: Number of blendshapes
            smoothing_method: Type of smoothing to apply
            alpha: Smoothing factor for exponential smoothing
            window_size: Window size for windowed smoothing methods
            learnable: Whether smoothing parameters are learnable
        """
        super().__init__()
        
        self.num_blendshapes = num_blendshapes
        self.smoothing_method = smoothing_method
        self.window_size = window_size
        self.learnable = learnable
        
        if learnable:
            # Learnable smoothing parameters
            if smoothing_method == "exponential":
                self.alpha = nn.Parameter(torch.tensor(alpha))
            elif smoothing_method == "gaussian":
                # Learnable Gaussian weights
                self.gaussian_weights = nn.Parameter(
                    torch.ones(window_size) / window_size
                )
        else:
            # Fixed smoothing parameters
            self.register_buffer('alpha', torch.tensor(alpha))
            if smoothing_method == "gaussian":
                # Fixed Gaussian weights
                weights = self._create_gaussian_weights(window_size)
                self.register_buffer('gaussian_weights', weights)
        
        # Buffer for maintaining state
        self.register_buffer('prev_output', torch.zeros(1, num_blendshapes))
        self.register_buffer('history', torch.zeros(window_size, 1, num_blendshapes))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
    
    def _create_gaussian_weights(self, window_size: int) -> torch.Tensor:
        """Create Gaussian weights for windowed smoothing."""
        x = torch.arange(window_size, dtype=torch.float32)
        center = (window_size - 1) / 2
        sigma = window_size / 6  # 3-sigma window
        
        weights = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        weights = weights / weights.sum()
        
        return weights
    
    def forward(
        self,
        blendshapes: torch.Tensor,
        reset_state: bool = False,
    ) -> torch.Tensor:
        """
        Apply temporal smoothing to blendshapes.
        
        Args:
            blendshapes: Input blendshapes of shape (B, 52)
            reset_state: Whether to reset internal state
            
        Returns:
            Smoothed blendshapes of shape (B, 52)
        """
        if reset_state:
            self._reset_state(blendshapes.device)
        
        batch_size = blendshapes.shape[0]
        
        if self.smoothing_method == "exponential":
            return self._exponential_smoothing(blendshapes)
        elif self.smoothing_method == "gaussian":
            return self._gaussian_smoothing(blendshapes)
        elif self.smoothing_method == "median":
            return self._median_smoothing(blendshapes)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")
    
    def _exponential_smoothing(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """Apply exponential moving average smoothing."""
        batch_size = blendshapes.shape[0]
        
        # Expand previous output to match batch size
        if self.prev_output.shape[0] != batch_size:
            self.prev_output = self.prev_output.expand(batch_size, -1).contiguous()
        
        # Apply exponential smoothing
        alpha = torch.sigmoid(self.alpha) if self.learnable else self.alpha
        smoothed = alpha * self.prev_output + (1 - alpha) * blendshapes
        
        # Update state (detach to avoid gradient issues)
        self.prev_output = smoothed.detach()
        
        return smoothed
    
    def _gaussian_smoothing(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian weighted smoothing over history window."""
        batch_size = blendshapes.shape[0]
        
        # Update history
        self._update_history(blendshapes)
        
        # Apply Gaussian weights
        if self.learnable:
            weights = F.softmax(self.gaussian_weights, dim=0)
        else:
            weights = self.gaussian_weights
        
        # Weighted sum over history
        weights = weights.view(-1, 1, 1)  # (window_size, 1, 1)
        history = self.history[:, :batch_size, :]  # (window_size, B, 52)
        
        smoothed = torch.sum(weights * history, dim=0)  # (B, 52)
        
        return smoothed
    
    def _median_smoothing(self, blendshapes: torch.Tensor) -> torch.Tensor:
        """Apply median filtering over history window."""
        batch_size = blendshapes.shape[0]
        
        # Update history
        self._update_history(blendshapes)
        
        # Compute median over history
        history = self.history[:, :batch_size, :]  # (window_size, B, 52)
        smoothed = torch.median(history, dim=0)[0]  # (B, 52)
        
        return smoothed
    
    def _update_history(self, blendshapes: torch.Tensor):
        """Update history buffer with new blendshapes."""
        batch_size = blendshapes.shape[0]
        
        # Ensure history buffer has correct batch size
        if self.history.shape[1] != batch_size:
            self.history = self.history[:, :1, :].expand(-1, batch_size, -1).contiguous()
        
        # Update circular buffer
        ptr = self.history_ptr.item()
        self.history[ptr] = blendshapes.detach()
        self.history_ptr = (ptr + 1) % self.window_size
    
    def _reset_state(self, device: torch.device):
        """Reset internal state buffers."""
        self.prev_output.zero_()
        self.history.zero_()
        self.history_ptr.zero_()
        
        # Move buffers to correct device
        self.prev_output = self.prev_output.to(device)
        self.history = self.history.to(device)
        self.history_ptr = self.history_ptr.to(device)


class BlendshapeConstraints(nn.Module):
    """
    Constraint module for enforcing blendshape validity and relationships.
    
    Applies physical constraints and regularization to ensure realistic
    blendshape combinations and smooth animations.
    """
    
    def __init__(
        self,
        num_blendshapes: int = 52,
        mutual_exclusions: Optional[list] = None,
        value_constraints: Optional[dict] = None,
        smoothness_weight: float = 0.1,
    ):
        """
        Initialize constraint module.
        
        Args:
            num_blendshapes: Number of blendshapes
            mutual_exclusions: List of mutually exclusive blendshape pairs
            value_constraints: Dictionary of value constraints per blendshape
            smoothness_weight: Weight for temporal smoothness constraint
        """
        super().__init__()
        
        self.num_blendshapes = num_blendshapes
        self.smoothness_weight = smoothness_weight
        
        # Mutual exclusion constraints (e.g., mouth open/closed)
        if mutual_exclusions is None:
            # Default ARKit exclusions (simplified)
            mutual_exclusions = [
                (25, 26),  # jawOpen vs jawLeft/Right
                (20, 21),  # mouthClose vs mouthFunnel
            ]
        
        self.exclusion_pairs = mutual_exclusions
        
        # Value constraints (min/max values per blendshape)
        if value_constraints is None:
            value_constraints = {}
        
        # Convert to tensors for efficiency
        self.min_values = torch.zeros(num_blendshapes)
        self.max_values = torch.ones(num_blendshapes)
        
        for bs_idx, (min_val, max_val) in value_constraints.items():
            self.min_values[bs_idx] = min_val
            self.max_values[bs_idx] = max_val
        
        self.register_buffer('min_values_buf', self.min_values)
        self.register_buffer('max_values_buf', self.max_values)
        
        # Previous state for smoothness
        self.register_buffer('prev_blendshapes', torch.zeros(1, num_blendshapes))
    
    def forward(
        self,
        blendshapes: torch.Tensor,
        apply_constraints: bool = True,
        return_violations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Apply constraints to blendshapes.
        
        Args:
            blendshapes: Input blendshapes of shape (B, 52)
            apply_constraints: Whether to apply hard constraints
            return_violations: Whether to return constraint violations
            
        Returns:
            Tuple of (constrained_blendshapes, violations_dict)
        """
        constrained = blendshapes.clone()
        violations = {} if return_violations else None
        
        # Value range constraints
        if apply_constraints:
            constrained = torch.clamp(
                constrained,
                min=self.min_values_buf,
                max=self.max_values_buf
            )
        
        if return_violations:
            violations['range_violations'] = (
                (blendshapes < self.min_values_buf) | 
                (blendshapes > self.max_values_buf)
            ).float().mean()
        
        # Mutual exclusion constraints
        for bs1, bs2 in self.exclusion_pairs:
            if apply_constraints:
                # Soft mutual exclusion using sigmoid
                combined = constrained[:, bs1] + constrained[:, bs2]
                constrained[:, bs1] = constrained[:, bs1] / (combined + 1e-8)
                constrained[:, bs2] = constrained[:, bs2] / (combined + 1e-8)
            
            if return_violations:
                overlap = torch.min(blendshapes[:, bs1], blendshapes[:, bs2])
                violations[f'exclusion_{bs1}_{bs2}'] = overlap.mean()
        
        # Temporal smoothness (as regularization, not hard constraint)
        if return_violations and self.prev_blendshapes.shape[0] == blendshapes.shape[0]:
            temporal_diff = torch.abs(blendshapes - self.prev_blendshapes)
            violations['temporal_smoothness'] = temporal_diff.mean()
        
        # Update previous state
        self.prev_blendshapes = blendshapes.detach()
        
        return constrained, violations
    
    def reset_state(self):
        """Reset internal state."""
        self.prev_blendshapes.zero_()


def validate_blendshape_output(blendshapes: torch.Tensor) -> dict:
    """
    Validate blendshape output for quality and consistency.
    
    Args:
        blendshapes: Blendshape coefficients of shape (B, 52)
        
    Returns:
        Dictionary with validation results
    """
    results = {'valid': True, 'warnings': [], 'stats': {}}
    
    if blendshapes.dim() != 2 or blendshapes.shape[1] != 52:
        results['valid'] = False
        results['warnings'].append(f"Expected shape (B, 52), got {blendshapes.shape}")
        return results
    
    # Check value range
    min_val = blendshapes.min().item()
    max_val = blendshapes.max().item()
    results['stats']['value_range'] = (min_val, max_val)
    
    if min_val < 0:
        results['warnings'].append(f"Negative values detected: {min_val:.3f}")
    if max_val > 1:
        results['warnings'].append(f"Values above 1 detected: {max_val:.3f}")
    
    # Check for NaN/inf
    if torch.isnan(blendshapes).any():
        results['valid'] = False
        results['warnings'].append("NaN values detected")
    
    if torch.isinf(blendshapes).any():
        results['valid'] = False
        results['warnings'].append("Infinite values detected")
    
    # Statistical analysis
    mean_activation = blendshapes.mean(dim=0)  # Per-blendshape mean
    std_activation = blendshapes.std(dim=0)    # Per-blendshape std
    
    results['stats']['mean_activation'] = mean_activation.mean().item()
    results['stats']['std_activation'] = std_activation.mean().item()
    results['stats']['active_blendshapes'] = (mean_activation > 0.1).sum().item()
    
    # Check for dead blendshapes (never activated)
    dead_blendshapes = (blendshapes.max(dim=0)[0] < 0.01).sum().item()
    results['stats']['dead_blendshapes'] = dead_blendshapes
    
    if dead_blendshapes > 10:
        results['warnings'].append(f"Many inactive blendshapes: {dead_blendshapes}/52")
    
    # Check for oversaturated blendshapes (always near max)
    saturated_blendshapes = (blendshapes.min(dim=0)[0] > 0.9).sum().item()
    results['stats']['saturated_blendshapes'] = saturated_blendshapes
    
    if saturated_blendshapes > 5:
        results['warnings'].append(f"Many saturated blendshapes: {saturated_blendshapes}/52")
    
    return results