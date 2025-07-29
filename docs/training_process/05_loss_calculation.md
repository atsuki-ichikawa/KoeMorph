# ステップ5: 時系列考慮型損失計算 (Temporal-Aware Loss Calculation)

## 概要
時系列モデルの予測と正解値の差を、時間的連続性を考慮しながら複数の観点から計算し、
Temporal smoothingを含む学習信号を生成します。

## 時系列損失計算の流れ

### 1. 損失関数の呼び出し
**実装場所**: `src/train_sequential.py:SequentialTrainer.train_step()`

```python
# 時系列ウィンドウの最終フレームを予測
pred_blendshapes = outputs['blendshapes']  # (B, 52)
target_frame = target_blendshapes[:, -1, :]  # (B, 52) - 最後のフレーム

loss = self.loss_fn(pred_blendshapes, target_frame)
```

**入力**:
- pred_blendshapes: `torch.Size([4, 52])` - 最終フレームの予測
- target_frame: `torch.Size([4, 52])` - 正解の最終フレーム

### 2. 時系列拡張KoeMorphLoss
**実装場所**: `src/model/losses.py` + Temporal smoothing components

**設定** (configs/dual_stream_config.yaml):
```yaml
loss:
  mse_weight: 1.0           # MSE損失の重み
  l1_weight: 0.1           # L1損失の重み  
  perceptual_weight: 0.5   # 知覚的損失の重み
  temporal_weight: 0.2     # 時間的連続性損失の重み
  smoothing_weight: 0.1    # Temporal smoothing損失の重み
```

### 3. 各損失成分の計算

#### 3.1 基本MSE損失 (Mean Squared Error)
**実装場所**: `src/model/losses.py:116-119`

```python
if self.mse_weight > 0:
    mse = F.mse_loss(pred_blendshapes, target_blendshapes)
    total_loss += self.mse_weight * mse  # 1.0 * mse
```

**計算式**:
```
MSE = (1/N) × Σ(pred - target)²
```

**時系列での特徴**:
- 最終フレームの精度を重視
- 時系列コンテキストからの予測精度を測定
- 値域: [0, ∞)

#### 3.2 L1損失 (Mean Absolute Error)
```python
if self.l1_weight > 0:
    l1 = F.l1_loss(pred_blendshapes, target_blendshapes)
    total_loss += self.l1_weight * l1  # 0.1 * l1
```

**時系列での利点**:
- 急激な表情変化に対してロバスト
- ノイズの多い時系列データに適応
- スパース性を促進（不必要な動きを抑制）

#### 3.3 Enhanced Perceptual損失
**実装場所**: 時系列専用の拡張版

```python
if self.perceptual_weight > 0:
    # デュアルストリーム用のperceptual loss
    
    # 口の動き専用ブレンドシェイプ（mel特徴主導）
    mouth_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # jawOpen, mouthなど
    mouth_pred = pred_blendshapes[:, mouth_indices]
    mouth_target = target_blendshapes[:, mouth_indices]
    
    # 表情専用ブレンドシェイプ（emotion特徴主導）
    expression_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # browRaise, eyeCloseなど
    expr_pred = pred_blendshapes[:, expression_indices]
    expr_target = target_blendshapes[:, expression_indices]
    
    # 重み付きperceptual loss
    mouth_loss = F.mse_loss(mouth_pred, mouth_target)
    expression_loss = F.mse_loss(expr_pred, expr_target)
    
    perceptual_loss = mouth_loss + expression_loss
    total_loss += self.perceptual_weight * perceptual_loss
```

#### 3.4 Temporal Smoothing損失（新規追加）
**実装場所**: モデル内のtemporal smoothing機構と連携

```python
if self.smoothing_weight > 0 and hasattr(model, 'prev_blendshapes'):
    # 前フレームとの差分を制約
    if model.prev_blendshapes is not None:
        # 平滑化後の値と平滑化前の値の差を測定
        smoothing_diff = F.l1_loss(
            pred_blendshapes,  # 平滑化後
            model.raw_prediction  # 平滑化前
        )
        
        # 適切な平滑化レベルを学習
        smoothing_loss = smoothing_diff
        total_loss += self.smoothing_weight * smoothing_loss
```

**目的**:
- 適応的な時間的平滑化を学習
- 急激な変化と滑らかな変化のバランス
- αパラメータの自動調整

#### 3.5 Sequential Consistency損失（時系列専用）
**実装場所**: 時系列training特有

```python
if self.temporal_weight > 0 and batch_idx > 0:
    # 同じファイル内の連続するバッチ間の一貫性
    if current_file_idx == prev_file_idx:
        # 前バッチの最終予測と現在の初期状態の一貫性
        consistency_loss = F.mse_loss(
            pred_blendshapes,  # 現在の予測
            model.prev_blendshapes.detach()  # 前フレームの状態
        )
        total_loss += self.temporal_weight * consistency_loss
```

**目的**:
- ファイル内でのフレーム間連続性を保証
- 時系列の滑らかな遷移を学習
- バッチ境界での不連続性を防止

### 4. 時系列総合損失の計算

**最終的な損失**:
```python
total_loss = (1.0 * mse + 
             0.1 * l1 + 
             0.5 * perceptual_loss +
             0.2 * temporal_consistency +
             0.1 * smoothing_loss)
```

**例（典型的な値）**:
- MSE: 0.02
- L1: 0.05  
- Perceptual: 0.03
- Temporal: 0.01
- Smoothing: 0.008
- **Total**: 1.0×0.02 + 0.1×0.05 + 0.5×0.03 + 0.2×0.01 + 0.1×0.008 = 0.0428

### 5. 時系列勾配計算への影響

#### 各損失成分の勾配特性

**MSE勾配**:
```
∂MSE/∂pred = 2 × (pred - target) / N
```
- 時系列コンテキストからの予測精度を重視
- 大きな誤差に対して二次的なペナルティ

**L1勾配**:
```
∂L1/∂pred = sign(pred - target) / N
```
- 急激な表情変化に対してロバスト
- ゼロ付近でのスパース性を促進

**Temporal Smoothing勾配**:
```
∂S/∂pred = α × ∂L1(pred, prev) / ∂pred
```
- 前フレームとの連続性を考慮
- 学習可能なα係数による適応的制御

## ストライド別損失重み調整

### Dense Sampling (stride=1)
```python
# 高密度学習時の重み調整
loss_weights = {
    'mse_weight': 1.0,
    'l1_weight': 0.1,
    'perceptual_weight': 0.5,
    'temporal_weight': 0.3,  # 時間的連続性を重視
    'smoothing_weight': 0.15
}
```

### Sparse Sampling (stride>1)
```python
# 疎サンプリング時の重み調整
loss_weights = {
    'mse_weight': 1.0,
    'l1_weight': 0.1,
    'perceptual_weight': 0.7,  # 意味的精度を重視
    'temporal_weight': 0.1,    # 連続性重視度を下げる
    'smoothing_weight': 0.05
}
```

## デバッグと可視化

### 時系列損失の監視
```python
# 各成分を個別に記録
losses = {
    "mse": mse.item(),
    "l1": l1.item(),
    "perceptual": perceptual_loss.item(),
    "temporal": temporal_consistency.item(),
    "smoothing": smoothing_loss.item(),
    "total": total_loss.item(),
    "stride": current_stride,
    "file_transition": is_new_file
}
```

### 時系列学習での正常な損失値範囲
- **学習初期** (stride=32): 0.15 - 0.6
- **中期** (stride=8): 0.08 - 0.25  
- **後期** (stride=1): 0.02 - 0.08
- **異常値**: > 1.0（発散の兆候）

### ファイル境界での損失パターン
```python
# ファイル境界での典型的なパターン
if is_file_boundary:
    # Temporal lossが一時的に増加（正常）
    expected_temporal_increase = True
    # Smoothing lossがリセット
    smoothing_loss_should_be_small = True
```

## 時系列損失関数の設計思想

1. **MSE**: 基本的な予測精度、時系列コンテキストの有効性
2. **L1**: 急激な表情変化への頑健性
3. **Perceptual**: デュアルストリーム（口vs表情）の役割分担
4. **Temporal**: フレーム間の自然な遷移
5. **Smoothing**: 適応的な時間的平滑化

## 学習段階別の損失戦略

### 初期段階 (Epoch 0-25)
- 粗いストライドで基本的なパターン学習
- MSEとPerceptual損失を重視
- Temporal損失は低めに設定

### 中期段階 (Epoch 25-75)  
- ストライドを段階的に減少
- Temporal損失の重みを増加
- 細かい時系列パターンの学習

### 後期段階 (Epoch 75-100)
- Dense sampling (stride=1)
- Smoothing損失を最大化
- 最終的な品質向上

## まとめ

時系列考慮型損失計算では：
1. **因果的予測**: 256フレームコンテキストから次フレームを予測
2. **デュアルストリーム対応**: 口と表情の特化損失
3. **適応的重み**: ストライドとエポックに応じた損失重み調整
4. **時間的連続性**: ファイル内でのフレーム間連続性を保証
5. **学習可能平滑化**: Temporal smoothingのパラメータを自動調整

次のステップでは、これらの損失からの勾配が時系列状態を保持しながらどのようにバックプロパゲーションされるかを見ていきます。