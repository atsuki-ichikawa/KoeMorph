# Technical Specifications and Mathematical Formulations

## System Architecture Specifications

### Input/Output Specifications
```
Input Audio:
- Sample Rate: 16kHz
- Format: Mono WAV
- Context Window: 8.5 seconds
- Frame Rate: 30fps or 60fps

Output Blendshapes:
- Format: 52 ARKit coefficients
- Range: [0, 1] continuous values
- Frame Rate: 30fps or 60fps (configurable)
- Output Type: Sequential time-series (B, T, 52)
```

### Model Architecture Parameters
```
Enhanced Mel Stream:
- Long-term Context: 8.5s (30fps: 256 frames, 60fps: 512 frames)
- Short-term Detail: 3 frames temporal concatenation
- Frequency Bins: 80 mel-scale bins
- Total Dimensions: 30fps: 20,720D, 60fps: 41,200D

Enhanced Emotion Stream:
- Context Windows: 3 × 20s (current, -300ms, -600ms)
- Raw Features: 88D eGeMAPS per window
- Concatenated: 264D (88 × 3)
- Compressed: 256D (learnable compression)

Cross-Attention:
- Query Dimension: 52 (ARKit blendshapes)
- Key/Value Dimension: 256 (d_model)
- Attention Heads: 8
- Temperature Parameter: τ ∈ [0.05, 0.5]
```

## Mathematical Formulations

### 1. Enhanced Dual-Stream Architecture

#### Information Density Ratio
```
Definition: R_info = D_mel / D_emotion

Baseline Configuration:
D_mel_baseline = 256 × 80 = 20,480
D_emotion_baseline = 88
R_baseline = 20,480 / 88 = 232.7

Enhanced Configuration:  
D_mel_enhanced = 256 × 80 + 3 × 80 = 20,720  (30fps)
D_mel_enhanced = 512 × 80 + 3 × 80 = 41,200  (60fps)
D_emotion_enhanced = 256 (compressed from 264)

R_enhanced_30fps = 20,720 / 256 = 80.9
R_enhanced_60fps = 41,200 / 256 = 160.9

Improvement Factor: η = R_baseline / R_enhanced = 232.7 / 80.9 = 2.87x
```

#### Multi-Frame Rate Dynamics  
```
Dynamic hop_length Calculation:
hop_length(fps) = ⌊sample_rate / fps⌋

For 16kHz audio:
hop_length(30) = ⌊16000 / 30⌋ = 533 samples (33.33ms)
hop_length(60) = ⌊16000 / 60⌋ = 267 samples (16.67ms)

Frame Count Calculation:
N_frames(fps, duration) = ⌊duration × fps⌋

Context Frames:
N_context(30fps, 8.5s) = ⌊8.5 × 30⌋ = 256 frames
N_context(60fps, 8.5s) = ⌊8.5 × 60⌋ = 512 frames
```

### 2. Enhanced Feature Extraction

#### Mel-Spectrogram Enhancement
```
Long-term Mel Features:
M_long = MelSpectrogram(audio, sr=16000, n_fft=1024, 
                       hop_length=hop_length(fps), n_mels=80)
Shape: (80, N_frames)

Short-term Temporal Detail:
M_short = M_long[:, -3:]  # Last 3 frames
Shape: (80, 3)

Enhanced Mel Concatenation:
M_enhanced = Concat([Reshape(M_long, (N_frames, 80)), 
                     Reshape(M_short, (3, 80))], axis=0)
Shape: (N_frames + 3, 80)

Total Information:
I_mel = N_frames × 80 + 3 × 80 = 80 × (N_frames + 3)
```

#### Emotion Feature Enhancement
```
3-Window eGeMAPS Extraction:
Let audio_t be audio at time t
E_current = eGeMAPS(audio_t[-20s:])           # Current window
E_past300 = eGeMAPS(audio_(t-0.3s)[-20s:])    # 300ms ago  
E_past600 = eGeMAPS(audio_(t-0.6s)[-20s:])    # 600ms ago

Each E_* ∈ ℝ^88

Concatenation:
E_concat = Concat([E_current, E_past300, E_past600])
Shape: (264,)

Learnable Compression:
E_compressed = Linear_256×264(E_concat) + bias_256
Shape: (256,)

Where Linear_256×264 ∈ ℝ^256×264 is a learnable transformation.
```

### 3. Cross-Attention Mechanism

#### Dual-Stream Cross-Attention
```
ARKit Blendshape Queries:
Q_arkit = Learnable_Parameter ∈ ℝ^52×d_model
where d_model = 256

Mel Stream Processing:
Input: M_enhanced ∈ ℝ^(N_frames+3)×80
K_mel, V_mel = Encoder_mel(M_enhanced) ∈ ℝ^(N_frames+3)×d_model

Frequency-wise Processing:
M_freq = Reshape(M_enhanced, (80, N_frames+3, 1))
K_mel_freq, V_mel_freq = Encoder_mel_freq(M_freq) ∈ ℝ^80×(N_frames+3)×d_model

Attention Computation:
A_mel = MultiHeadAttention(Q_arkit, K_mel_freq, V_mel_freq)
A_mel ∈ ℝ^52×d_model

Emotion Stream Processing:
Input: E_compressed ∈ ℝ^256
K_emotion, V_emotion = Encoder_emotion(E_compressed) ∈ ℝ^1×d_model
A_emotion = MultiHeadAttention(Q_arkit, K_emotion, V_emotion)
A_emotion ∈ ℝ^52×d_model
```

#### Natural Specialization Learning
```
Learnable Stream Weights:
θ_mel ∈ ℝ^52     # Mel stream weights per blendshape
θ_emotion ∈ ℝ^52  # Emotion stream weights per blendshape

Temperature-Controlled Softmax:
w_mel = softmax(θ_mel / τ)
w_emotion = softmax(θ_emotion / τ)

where τ > 0 is the temperature parameter controlling specialization sharpness.

Weighted Fusion:
A_fused = w_mel ⊙ A_mel + w_emotion ⊙ A_emotion
where ⊙ denotes element-wise multiplication.

Final Blendshape Prediction:
B_raw = Decoder(A_fused) ∈ ℝ^52
B_output = σ(B_raw)  # Sigmoid to ensure [0,1] range
```

### 4. Sequential Processing

#### Sliding Window Mechanism
```
For input audio of length L samples:
N_total_frames = ⌊L / hop_length⌋

Window Parameters:
W = N_frames  # Window size (256 for 30fps, 512 for 60fps)  
S = stride_frames  # Stride (configurable, default=1)

Number of Output Windows:
N_windows = max(1, ⌊(N_total_frames - W) / S⌋ + 1)

Window Extraction:
For i ∈ [0, N_windows):
    start_frame = i × S
    end_frame = start_frame + W
    
    start_sample = start_frame × hop_length
    end_sample = end_frame × hop_length
    
    audio_window_i = audio[start_sample:end_sample]
```

#### Efficient Emotion Processing
```
Traditional Approach (inefficient):
For each window i:
    E_i = ExtractEmotion(audio_window_i)  # Redundant computation

Our Approach (efficient):
E_global = ExtractEmotion(audio_full)  # Single extraction
For each window i:
    E_i = E_global  # Reuse global emotion features

Memory Reduction:
Memory_traditional = N_windows × Memory_emotion_extraction
Memory_ours = 1 × Memory_emotion_extraction
Efficiency_gain = N_windows (typically 100-300x)
```

#### Temporal Smoothing
```
Learnable Smoothing Parameter:
α ∈ [0, 1], initialized as α = sigmoid(α_raw) where α_raw ~ N(0, 0.1)

Temporal Smoothing Update:
B_t^smooth = α × B_t^raw + (1 - α) × B_{t-1}^smooth

where:
- B_t^raw: Raw prediction at time t
- B_{t-1}^smooth: Previous smoothed prediction
- B_t^smooth: Current smoothed output

State Management:
At file boundaries: B_{-1}^smooth = B_0^raw (reset state)
Within file: Maintain temporal continuity
```

### 5. Loss Functions

#### Multi-Component Loss
```
Total Loss:
L_total = w_mse × L_mse + w_l1 × L_l1 + w_perceptual × L_perceptual + 
          w_temporal × L_temporal + w_smoothing × L_smoothing

where w_* are loss weights.

MSE Loss:
L_mse = 1/N ∑_{i=1}^N ||B_pred^i - B_true^i||_2^2

L1 Loss:  
L_l1 = 1/N ∑_{i=1}^N ||B_pred^i - B_true^i||_1

Perceptual Loss (Stream Specialization):
L_perceptual = w_mouth × L_mouth + w_expression × L_expression

where:
L_mouth = MSE(B_pred[mouth_indices], B_true[mouth_indices])
L_expression = MSE(B_pred[expression_indices], B_true[expression_indices])

Temporal Consistency Loss:
L_temporal = 1/(N-1) ∑_{i=1}^{N-1} ||∇B_pred^i - ∇B_true^i||_2^2

where ∇B^i = B^{i+1} - B^i (temporal gradient)

Smoothing Regularization:
L_smoothing = ||α - α_target||_2^2

where α_target is the desired smoothing level (typically 0.1-0.2).
```

### 6. Performance Metrics

#### Real-Time Factor (RTF)
```
RTF = T_processing / T_audio

where:
- T_processing: Time taken for processing
- T_audio: Duration of input audio

Real-time constraint: RTF < 1.0
High-performance target: RTF < 0.1
```

#### Information Balance Metrics
```
Stream Specialization Ratio:
For blendshape category C (mouth, expression, etc.):

Specialization_C = |w_mel^C - w_emotion^C| / (w_mel^C + w_emotion^C)

where w_mel^C, w_emotion^C are average weights for category C.

Overall Specialization:
S_overall = 1/|Categories| ∑_{C} Specialization_C

Good specialization: S_overall ∈ [0.6, 0.9]
```

#### Memory Efficiency
```
Memory Components:
M_mel_buffer = context_frames × 80 × 4 bytes
M_emotion_buffer = 20s × 16000 × 4 bytes = 1.28 MB
M_model_params = |θ| × 4 bytes ≈ 8.2 MB
M_intermediate = batch_size × max_sequence_length × 52 × 4 bytes

Total Memory:
M_total = M_mel_buffer + M_emotion_buffer + M_model_params + M_intermediate

For 30fps: M_total ≈ 355 MB
For 60fps: M_total ≈ 450 MB
```

## Implementation Constants

```python
# Audio Processing
SAMPLE_RATE = 16000
N_FFT = 1024
N_MELS = 80
FMIN = 80.0
FMAX = 8000.0

# Model Architecture  
D_MODEL = 256
N_HEADS = 8
N_BLENDSHAPES = 52
CONTEXT_DURATION = 8.5  # seconds

# Training Parameters
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
SCHEDULER = "cosine_annealing"

# Loss Weights
W_MSE = 1.0
W_L1 = 0.1
W_PERCEPTUAL = 0.5
W_TEMPORAL = 0.2
W_SMOOTHING = 0.1

# Performance Targets
RTF_TARGET = 0.1
MEMORY_TARGET_MB = 500
MODEL_SIZE_TARGET_MB = 10
```

## Hardware Requirements

### Minimum Requirements
```
GPU: GTX 1060 or equivalent (4GB VRAM)
CPU: Intel i5-8400 or equivalent
RAM: 8GB
Storage: 2GB available space
```

### Recommended Requirements
```
GPU: RTX 3070 or equivalent (8GB VRAM)
CPU: Intel i7-10700K or equivalent  
RAM: 16GB
Storage: 5GB available space (including datasets)
```

### Mobile Deployment
```
iOS: iPhone 12 Pro or newer (A14 Bionic+)
Android: Snapdragon 888 or equivalent
RAM: 6GB minimum, 8GB recommended
Storage: 1GB available space
```

This technical specification provides comprehensive mathematical formulations and implementation details suitable for reproducible research and academic publication.