# ステップ4: デュアルストリームフォワードパス (Dual-Stream Forward Pass)

## 概要
デュアルストリームアーキテクチャでは、mel-spectrogramとemotion特徴を独立したストリームで処理し、
自然な特化を学習しながらARKitブレンドシェイプ値を予測します。
Multi-Frame Rate対応により30fps/60fpsでの推論が可能で、SequentialDualStreamModelでは
完全な時系列出力をサポートします。

## フォワードパスの流れ

### 1. モデル呼び出し（Multi-Frame Rate & Sequential対応）

#### 1.1 Sequential Training Model
**実装場所**: `src/train_sequential.py:SequentialTrainer.train_epoch()`

```python
# 時系列状態の管理
if file_idx != self.current_file_idx:
    self.model.reset_temporal_state()  # ファイル境界でリセット
    
# SequentialDualStreamModel の使用（推奨）
outputs = self.model(audio, return_attention=False)
pred_blendshapes = outputs['blendshapes']  # (B, T_out, 52) - 完全時系列出力
```

#### 1.2 基本 Dual-Stream Model
**実装場所**: `src/model/simplified_dual_stream_model.py`

```python
# 単一フレーム出力（legacy）
outputs = self.model(audio, return_attention=False)
pred_blendshapes = outputs['blendshapes']  # (B, 52) - 単一フレーム
```

**入力（Multi-Frame Rate対応）**: 
- **30fps**: audio: `torch.Size([B, 136576])` (8.5秒分の音声, hop_length=533)
- **60fps**: audio: `torch.Size([B, 136576])` (8.5秒分の音声, hop_length=267)

**出力**:
- **Sequential Model**: `(B, T_out, 52)` - 完全時系列（T_out = 入力音声長に応じた可変長）
- **Basic Model**: `(B, 52)` - 単一フレーム

### 2. デュアルストリーム特徴抽出（Multi-Frame Rate対応）
**実装場所**: `src/model/simplified_dual_stream_model.py`, `src/model/sequential_dual_stream_model.py`

#### 2.1 Enhanced Mel特徴抽出 (高頻度更新)
```python
# extract_mel_features() - Dynamic frame support
mel_features, mel_temporal_features = self.extract_mel_features(audio)
# 30fps: (B, 256, 80), 60fps: (B, 512, 80)
# temporal: (B, 3, 80) - 両モード共通
```

**処理の詳細（Multi-Frame Rate対応）**:
#### 30fpsモード:
- **コンテキスト**: 8.5秒 (256フレーム)
- **更新頻度**: 33.3ms (毎フレーム)
- **hop_length**: 533サンプル

#### 60fpsモード:
- **コンテキスト**: 8.5秒 (512フレーム)
- **更新頻度**: 16.7ms (毎フレーム)
- **hop_length**: 267サンプル

**共通**:
- **特徴次元**: 80 mel bins + 3フレーム時間詳細
- **総情報量**: 30fps: 20,720次元, 60fps: 41,200次元

#### 2.2 Enhanced Emotion特徴抽出 (長期コンテキスト)
```python
# extract_emotion_features() - Single extraction for efficiency
emotion_features, metadata = self.extract_emotion_features(audio)  # (B, 256)
```

**Enhanced OpenSMILE 3窓連結処理**:
- **コンテキスト**: 20秒×3窓 (現在, -300ms, -600ms)
- **更新頻度**: 300ms (30fps: ~9フレーム, 60fps: ~18フレーム)
- **特徴次元**: 256次元 (88×3→256圧縮)
- **Sequential効率化**: 全音声で単一抽出（従来は窓ごと抽出）

### 3. 特徴アライメント（Dynamic Frame Count対応）
```python
# align_features() - Frame count varies by fps
mel_features, emotion_features = self.align_features(mel_features, emotion_features)
# 30fps: 両方とも (B, 256, d_feature) にアライン
# 60fps: 両方とも (B, 512, d_feature) にアライン
```

**Dynamic Alignment**:
- **30fps**: 256フレームにアライメント
- **60fps**: 512フレームにアライメント
- **Emotion補間**: 256次元emotion特徴を対象フレーム数に線形補間

### 4. デュアルストリームクロスアテンション
**実装場所**: `src/model/dual_stream_attention.py:DualStreamCrossAttention`

```python
output = self.dual_stream_attention(
    mel_features=mel_features,        # (B, 256, 80)
    emotion_features=emotion_features, # (B, 256, 88)
    return_attention=False
)
```

#### 4.1 周波数軸分割 (Mel Stream)
```python
# Melを周波数軸で分割
mel_features = mel_features.transpose(1, 2)  # (B, 80, 256)
mel_features = mel_features.reshape(B * 80, 256, 1)
# 80個の独立したチャンネルとして処理
```

**理由**: 各周波数帯域が独立してアテンションパターンを学習

#### 4.2 並列ストリーム処理
```python
# Mel Stream (口の動きに特化)
mel_attended = self.mel_attention(
    self.blendshape_queries,  # (52, 256)
    mel_encoded,              # (B*80, 256, d_model)
    mel_encoded
)

# Emotion Stream (表情全体に特化)
emotion_attended = self.emotion_attention(
    self.blendshape_queries,  # (52, 256)
    emotion_encoded,          # (B, T_e, d_model)
    emotion_encoded
)
```

#### 4.3 自然な特化の学習
```python
# 学習可能な重み（強制的な3x係数を削除）
normalized_mel_weights = F.softmax(self.mel_weights / self.temperature, dim=0)
normalized_emotion_weights = F.softmax(self.emotion_weights / self.temperature, dim=0)

# 重み付き結合
final_output = (
    normalized_mel_weights.unsqueeze(0) * mel_attended +
    normalized_emotion_weights.unsqueeze(0) * emotion_attended
)
```

**重要な変更点**:
- 3倍の特化係数を削除
- モデルが自然に口/表情の役割分担を学習
- 温度パラメータで特化の鋭さを制御

### 5. Temporal Smoothing
**実装場所**: `src/model/simplified_dual_stream_model.py:apply_temporal_smoothing()`

```python
# 前フレームとの平滑化
if self.prev_blendshapes is not None:
    alpha = torch.sigmoid(self.smoothing_alpha)  # 学習可能
    smoothed = alpha * blendshapes + (1 - alpha) * self.prev_blendshapes
    self.prev_blendshapes = smoothed.detach()
else:
    self.prev_blendshapes = blendshapes.detach()
```

**時系列連続性**:
- ファイル内では状態を保持
- 急激な変化を抑制
- αは学習により最適化

## データフローまとめ（Sequential & Multi-Frame Rate対応）

### Sequential Model Data Flow:
```
入力音声 (B, 136576) [8.5秒]
    ├─→ Enhanced Mel特徴抽出 (一度で全窓)
    │   ↓
    │   30fps: Mel特徴 (B, 256, 80) + 時間詳細 (B, 3, 80)
    │   60fps: Mel特徴 (B, 512, 80) + 時間詳細 (B, 3, 80)
    │   ↓
    │   スライディング窓処理 (stride_frames間隔)
    │   ↓
    │   各窓で Dual-Stream Attention
    │   ↓
    │   Sequential Blendshapes (B, T_out, 52)
    │
    └─→ Enhanced Emotion特徴抽出 [効率的単一抽出]
        ↓
        3窓連結 Emotion特徴 (B, 256)
        ↓
        全窓で再利用（メモリ効率化）
        ↓
        Temporal Smoothing (窓間連続性)
        ↓
        最終Sequential出力 (B, T_out, 52)
```

### Basic Model Data Flow:
```
入力音声 (B, 136576) [8.5秒] → 単一フレーム出力 (B, 52)
```

**主要改善点**:
1. **Sequential Processing**: 完全な時系列出力
2. **Efficient Emotion Extraction**: 単一抽出で全窓カバー
3. **Multi-Frame Rate**: 30fps/60fps dynamic support
4. **Memory Optimization**: 固定バッファサイズ維持

## パラメータ数とメモリ

### パラメータ数（概算）
- Mel Encoder: 80×256 = 20,480
- Emotion Encoder: 88×256 = 22,528
- Blendshape Queries: 52×256 = 13,312
- Mel Attention: 256×256×3 = 196,608
- Emotion Attention: 256×256×3 = 196,608
- Stream Weights: 52×2 = 104
- Decoder: 256×128 + 128×52 = 39,424
- **合計**: 約489K パラメータ

### メモリ使用量（推定）
- Mel sliding window buffer: 8.5s × 16kHz × 4bytes = 544KB/stream
- Emotion buffer: 20s × 16kHz × 4bytes = 1.28MB/stream
- 中間活性化: 約20MB（バッチサイズ4）

## 主要な改善点

1. **独立したストリーム処理**
   - Melは高頻度・短期的な口の動き
   - Emotionは低頻度・長期的な表情

2. **自然な特化学習**
   - 強制的な役割分担を削除
   - データから最適な分担を学習

3. **時系列連続性**
   - Temporal smoothingで滑らかな動き
   - ファイル単位での状態管理

4. **効率的な処理**
   - 固定サイズウィンドウでメモリ一定
   - リアルタイム推論可能（RTF < 0.1）

## まとめ

デュアルストリームフォワードパス（Sequential & Multi-Frame Rate対応）では：
1. **Multi-Frame Rate対応**: 30fps（33.3ms）/60fps（16.7ms）での動的処理
2. **Sequential Processing**: 完全時系列出力による単一フレーム制限の解決
3. **Enhanced特徴抽出**: Mel長期+短期詳細、Emotion 3窓連結で情報バランス最適化
4. **効率的メモリ使用**: Emotion単一抽出による大幅なメモリ削減
5. **Dynamic Alignment**: フレームレートに応じた特徴アライメント（256/512フレーム）
6. **周波数軸分割**: Melの各周波数帯が独立してアテンション学習
7. **自然な重み学習**: 各ストリームが適切に特化（情報比率80.9:1維持）
8. **Temporal smoothing**: 時系列の連続性を保持（窓間とフレーム間）
9. **固定メモリ使用量**: 任意長の音声を一定メモリで処理可能

**パフォーマンス**:
- **30fps**: RTF ~0.06, メモリ ~355MB
- **60fps**: RTF ~0.08, メモリ ~450MB（両方ともリアルタイム対応）

次のステップでは、時系列を考慮した損失計算について説明します。