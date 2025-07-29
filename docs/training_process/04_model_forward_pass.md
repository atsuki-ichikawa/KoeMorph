# ステップ4: デュアルストリームフォワードパス (Dual-Stream Forward Pass)

## 概要
デュアルストリームアーキテクチャでは、mel-spectrogramとemotion特徴を独立したストリームで処理し、
自然な特化を学習しながらARKitブレンドシェイプ値を予測します。

## フォワードパスの流れ

### 1. モデル呼び出し
**実装場所**: `src/train_sequential.py:SequentialTrainer.train_epoch()`

```python
# 時系列状態の管理
if file_idx != self.current_file_idx:
    self.model.reset_temporal_state()  # ファイル境界でリセット
    
# フォワードパス
outputs = self.model(audio, return_attention=False)
pred_blendshapes = outputs['blendshapes']  # (B, 52)
```

**入力**: 
- audio: `torch.Size([B, 136576])` (8.5秒分の音声, B=バッチサイズ)

### 2. デュアルストリーム特徴抽出
**実装場所**: `src/model/simplified_dual_stream_model.py`

#### 2.1 Mel特徴抽出 (高頻度更新)
```python
# extract_mel_features()
mel_features = self.extract_mel_features(audio)  # (B, 256, 80)
```

**処理の詳細**:
- **コンテキスト**: 8.5秒 (256フレーム)
- **更新頻度**: 33.3ms (毎フレーム)
- **特徴次元**: 80 mel bins
- **hop_length**: 533サンプル

#### 2.2 Emotion特徴抽出 (長期コンテキスト)
```python
# extract_emotion_features()
emotion_features, metadata = self.extract_emotion_features(audio)
```

**OpenSMILE eGeMAPSの処理**:
- **コンテキスト**: 20秒 (前の処理も含む)
- **更新頻度**: 300ms (~9フレームごと)
- **特徴次元**: 88次元 (prosodic/paralinguistic features)
- **スライディングウィンドウ**: 過去の音声も保持

### 3. 特徴アライメント
```python
# align_features()
mel_features, emotion_features = self.align_features(mel_features, emotion_features)
# 両方とも (B, 256, d_feature) にアライン
```

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

## データフローまとめ

```
入力音声 (B, 136576) [8.5秒]
    ├─→ Mel特徴抽出
    │   ↓
    │   Mel特徴 (B, 256, 80)
    │   ↓
    │   周波数軸分割 (B*80, 256, 1)
    │   ↓
    │   Mel Encoding (B*80, 256, 256)
    │   ↓
    │   Mel Cross-Attention
    │   ↓
    │   Mel出力 (B, 52, 256)
    │
    └─→ Emotion特徴抽出 [20秒コンテキスト]
        ↓
        Emotion特徴 (B, T_e, 88)
        ↓
        時間軸アライメント
        ↓
        Emotion Encoding (B, 256, 256)
        ↓
        Emotion Cross-Attention
        ↓
        Emotion出力 (B, 52, 256)
        
    両ストリームの出力
        ↓
    自然な重み付け結合
        ↓
    最終特徴 (B, 52, 256)
        ↓
    Blendshapeデコーダ
        ↓
    生Blendshapes (B, 52)
        ↓
    Temporal Smoothing
        ↓
    最終出力 (B, 52)
```

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

デュアルストリームフォワードパスでは：
1. Mel特徴（8.5秒、33.3ms更新）とEmotion特徴（20秒、300ms更新）を並列抽出
2. 周波数軸分割により、Melの各周波数帯が独立してアテンション学習
3. 自然な重み学習により、各ストリームが適切に特化
4. Temporal smoothingにより時系列の連続性を保持
5. 固定メモリ使用量で任意長の音声を処理可能

次のステップでは、時系列を考慮した損失計算について説明します。