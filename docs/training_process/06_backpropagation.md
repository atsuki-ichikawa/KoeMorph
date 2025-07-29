# ステップ6: 時系列状態保持バックプロパゲーション (Sequential Backpropagation)

## 概要
時系列モデルの損失値から各パラメータの勾配を計算し、temporal smoothingの状態を保持しながら
デュアルストリームアーキテクチャ全体に勾配を伝播します。

## 時系列バックプロパゲーションの流れ

### 1. 勾配計算の開始
**実装場所**: `src/train_sequential.py:SequentialTrainer.train_step()`

```python
# 時系列考慮型損失からの勾配計算
loss.backward()  # 自動微分により勾配を計算
```

**前提条件**:
- loss: スカラー値 (例: 0.0428) - 時系列成分を含む複合損失
- temporal state (prev_blendshapes) を保持
- 計算グラフがデュアルストリーム全体をカバー

### 2. デュアルストリーム勾配伝播

#### 2.1 時系列損失から出力層への勾配
**複合損失の勾配**: `∂loss/∂pred_blendshapes`

```python
# 基本MSE成分の勾配
grad_mse = 2 * (pred_blendshapes - target_blendshapes) / batch_size

# L1成分の勾配  
grad_l1 = 0.1 * torch.sign(pred_blendshapes - target_blendshapes) / batch_size

# デュアルストリーム対応Perceptual勾配
grad_mouth = 0.5 * ∂mouth_loss/∂pred_blendshapes  # 口関連ブレンドシェイプ
grad_expr = 0.5 * ∂expression_loss/∂pred_blendshapes  # 表情関連ブレンドシェイプ

# Temporal smoothing勾配
grad_temporal = 0.1 * ∂smoothing_loss/∂pred_blendshapes  # 前フレームとの連続性

# 総合勾配 (時系列考慮型)
grad_output = (1.0 * grad_mse + 0.1 * grad_l1 + 
               0.5 * (grad_mouth + grad_expr) + 
               0.2 * grad_consistency + 0.1 * grad_temporal)
# Shape: [4, 52]
```

#### 2.2 Temporal Smoothing勾配の特殊処理
**実装場所**: `src/model/simplified_dual_stream_model.py:apply_temporal_smoothing()`

```python
# 学習可能な平滑化係数αの勾配
∂loss/∂α = ∂loss/∂smoothed * ∂smoothed/∂α
where: smoothed = α * current + (1 - α) * prev

# αの勾配計算
alpha_grad = grad_output * (current - prev_blendshapes.detach())
# prev_blendshapesはdetach()されているため勾配は流れない
```

**重要なポイント**:
- Temporal stateは勾配計算に参加するが、過去への勾配伝播は行わない
- αパラメータは学習可能で、適応的平滑化を実現

#### 2.3 デュアルストリームクロスアテンション勾配
**実装場所**: `src/model/dual_stream_attention.py:DualStreamCrossAttention`

```python
# 自然な重み学習の勾配
∂loss/∂mel_weights = grad_output * mel_attended / temperature
∂loss/∂emotion_weights = grad_output * emotion_attended / temperature

# 各ストリームのアテンション勾配
# Melストリーム (周波数軸分割対応)
∂loss/∂mel_attention → ∂loss/∂mel_encoded (B*80, 256, d_model)
∂loss/∂mel_encoded → reshape → ∂loss/∂mel_features (B, 80, 256)

# Emotionストリーム  
∂loss/∂emotion_attention → ∂loss/∂emotion_encoded (B, 256, d_model)
∂loss/∂emotion_encoded → ∂loss/∂emotion_features (B, 256, 88)
```

**周波数軸分割の勾配処理**:
```python
# 80個の独立チャンネルから元の形状へ復元
mel_grad_reshaped = mel_grad.view(B, 80, 256, d_model)
mel_features_grad = mel_grad_reshaped.sum(dim=3)  # d_model次元を集約
mel_features_grad = mel_features_grad.transpose(1, 2)  # (B, 256, 80)
```

#### 2.4 デュアルストリーム特徴抽出の勾配
**Melストリームの勾配**:
```python
# Mel sliding windowは微分不可（Librosa処理）
∂loss/∂mel_features → 勾配はここで停止
# 音声→Mel変換は非微分可能
```

**Emotionストリームの勾配**:
```python
# OpenSMILE eGeMAPSも微分不可
∂loss/∂emotion_features → 勾配はここで停止
# 音声→eGeMAPS変換は非微分可能
```

**特徴レベルでの学習**:
- 音声信号レベルでの勾配伝播は行わない
- 特徴抽出後のエンコーダ、アテンション、デコーダのみ学習
- End-to-endではなくmid-to-endの学習

### 3. 時系列モデルパラメータごとの勾配

#### デュアルストリーム主要パラメータと勾配の大きさ

```python
# 時系列モデルの勾配確認
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

**典型的な値（時系列学習）**:
- `smoothing_alpha.grad`: 0.0001-0.001 (Temporal smoothing係数)
- `mel_weights.grad`: 0.001-0.01 (Melストリーム重み)
- `emotion_weights.grad`: 0.001-0.01 (Emotionストリーム重み)
- `dual_stream_attention.mel_attention.*.grad`: 0.01-0.1
- `dual_stream_attention.emotion_attention.*.grad`: 0.005-0.05
- `blendshape_queries.grad`: 0.001-0.01
- `decoder.*.grad`: 0.01-0.1 (出力層に近い)

#### ストライド別勾配パターン

**Dense Sampling (stride=1)**:
```python
# 高密度学習時の勾配特性
temporal_grad_norm = 0.005-0.02  # 時間的連続性重視
mel_grad_norm = 0.01-0.05       # 細かい口の動き
emotion_grad_norm = 0.005-0.02  # 長期表情パターン
```

**Sparse Sampling (stride>1)**:
```python
# 疎サンプリング時の勾配特性
temporal_grad_norm = 0.001-0.005  # 連続性よりも意味重視
mel_grad_norm = 0.015-0.06        # 大きな変化を捉える
emotion_grad_norm = 0.01-0.04     # 表情の大きな変化
```

### 4. 時系列特有の勾配問題と対策

#### 4.1 Temporal State勾配の切断
**問題**: 長期依存による勾配爆発
**対策**: `prev_blendshapes.detach()`で過去への勾配を切断

```python
# 前フレーム状態は勾配計算に参加するが、過去に伝播しない
if self.prev_blendshapes is not None:
    alpha = torch.sigmoid(self.smoothing_alpha)
    smoothed = alpha * blendshapes + (1 - alpha) * self.prev_blendshapes.detach()
```

#### 4.2 ファイル境界での勾配リセット
**実装**: 新しいファイル開始時の勾配処理

```python
if file_idx != self.current_file_idx:
    # Temporal stateをリセット（勾配も切断）
    model.reset_temporal_state()
    # 境界での異常な勾配を防止
```

#### 4.3 デュアルストリーム勾配バランス
**問題**: MelとEmotionストリームの勾配不均衡
**対策**: 適応的重み正規化

```python
# 自然な重み学習での勾配正規化
mel_weight_grad = F.normalize(mel_weights.grad, dim=0)
emotion_weight_grad = F.normalize(emotion_weights.grad, dim=0)
```

### 5. 時系列学習のメモリ効率

#### 勾配とTemporal Stateのメモリ使用
```python
# パラメータ勾配: 各パラメータと同じサイズ
gradient_memory = param_count * 4 bytes

# Temporal state (detachされた前フレーム)
temporal_state_memory = batch_size * 52 * 4 bytes  # (4, 52)

# スライディングウィンドウバッファ
mel_buffer_memory = 256 * 80 * 4 bytes  # Melバッファ
emotion_buffer_memory = (20 * 30) * 88 * 4 bytes  # Emotionバッファ

# 総メモリ（バッチサイズ4）: 約15-20MB
```

#### 計算グラフの管理
```python
# 時系列では長期依存を避けるため、フレーム単位でグラフを切断
# Temporal smoothingのdetach()により、過去への勾配伝播を防止
```

### 6. 時系列デバッグ用勾配チェック

```python
def check_temporal_gradients(model, batch_idx, file_idx):
    """時系列特有の勾配チェック"""
    
    # 基本勾配チェック
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: {name} has no gradient!")
        elif param.grad.norm() == 0:
            print(f"WARNING: {name} has zero gradient!")
        elif param.grad.norm() > 10:
            print(f"WARNING: {name} has large gradient: {param.grad.norm()}")
    
    # 時系列特有のチェック
    if hasattr(model, 'smoothing_alpha'):
        alpha_grad = model.smoothing_alpha.grad
        print(f"Smoothing alpha gradient: {alpha_grad.item():.6f}")
    
    # ストリーム重みの勾配バランス
    if hasattr(model, 'dual_stream_attention'):
        mel_grad = model.dual_stream_attention.mel_weights.grad.norm()
        emotion_grad = model.dual_stream_attention.emotion_weights.grad.norm()
        balance_ratio = mel_grad / (emotion_grad + 1e-8)
        print(f"Stream gradient balance (Mel/Emotion): {balance_ratio:.3f}")
        
        # 理想的には0.5-2.0の範囲
        if balance_ratio < 0.1 or balance_ratio > 10.0:
            print(f"WARNING: Unbalanced stream gradients!")
```

## デュアルストリーム勾配フローの可視化

```
時系列複合損失 (スカラー: 0.0428)
    ↓ ∂loss/∂output (時系列成分含む)
予測ブレンドシェイプ (4, 52)
    ↓ Temporal smoothing勾配分岐
    ├─→ ∂loss/∂smoothing_alpha (学習可能α)
    └─→ デコーダ.backward()
最終特徴 (4, 52, 256)
    ↓ デュアルストリーム勾配分岐
    ├─→ Melストリーム勾配
    │   ├─→ ∂loss/∂mel_weights (自然特化重み)
    │   ├─→ Mel Attention.backward()
    │   └─→ Mel Features (4, 256, 80) [周波数軸分割考慮]
    │
    └─→ Emotionストリーム勾配  
        ├─→ ∂loss/∂emotion_weights (自然特化重み)
        ├─→ Emotion Attention.backward()
        └─→ Emotion Features (4, 256, 88) [長期コンテキスト]
    
    ↓ (特徴抽出は微分不可)
入力音声 (4, 136576) [勾配なし]
```

## まとめ

時系列状態保持バックプロパゲーションでは：
1. **複合損失の勾配分解**: MSE, L1, Perceptual, Temporal, Smoothingの各成分
2. **Temporal state管理**: 前フレームとの連続性を保ちつつ勾配爆発を防止
3. **デュアルストリーム勾配**: MelとEmotionの独立した勾配経路
4. **特化重み学習**: 自然な役割分担のための重み勾配
5. **効率的なメモリ管理**: ファイル境界での状態リセットと勾配切断
6. **適応的平滑化**: 学習可能αパラメータによる時間的制約の自動調整

次のステップでは、これらの勾配を使って時系列状態を維持しながらパラメータを更新する方法を見ていきます。