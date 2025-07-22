# ステップ5: 損失計算 (Loss Calculation)

## 概要
予測されたブレンドシェイプ値と正解値の差を複数の観点から計算し、学習信号を生成します。

## 損失計算の流れ

### 1. 損失関数の呼び出し
**実装場所**: `src/train.py:185`

```python
loss = self.loss_fn(pred_blendshapes, target_blendshapes)
```

**入力**:
- pred_blendshapes: `torch.Size([16, 52])` - モデルの予測
- target_blendshapes: `torch.Size([16, 52])` - 正解データ

### 2. KoeMorphLoss クラス
**実装場所**: `src/model/losses.py:87-182`

**設定** (configs/training/default.yaml):
```yaml
loss:
  mse_weight: 1.0        # MSE損失の重み
  l1_weight: 0.1         # L1損失の重み  
  perceptual_weight: 0.5 # 知覚的損失の重み
```

### 3. 各損失成分の計算

#### 3.1 MSE損失 (Mean Squared Error)
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

**特徴**:
- 大きな誤差により重いペナルティ
- 微分可能で最適化が安定
- 値域: [0, ∞)

**例**: 
- pred=[0.3, 0.7], target=[0.2, 0.8]
- MSE = ((0.3-0.2)² + (0.7-0.8)²) / 2 = 0.01

#### 3.2 L1損失 (Mean Absolute Error)
**実装場所**: `src/model/losses.py:124-127`

```python
if self.l1_weight > 0:
    l1 = F.l1_loss(pred_blendshapes, target_blendshapes)
    total_loss += self.l1_weight * l1  # 0.1 * l1
```

**計算式**:
```
L1 = (1/N) × Σ|pred - target|
```

**特徴**:
- 外れ値に対してロバスト
- スパース性を促進
- 値域: [0, ∞)

**例**:
- pred=[0.3, 0.7], target=[0.2, 0.8]
- L1 = (|0.3-0.2| + |0.7-0.8|) / 2 = 0.1

#### 3.3 Perceptual損失
**実装場所**: `src/model/losses.py:130-146`

```python
if self.perceptual_weight > 0:
    # アクティブなブレンドシェイプの違いをペナルティ
    pred_active = (pred_blendshapes > 0.1).float()
    target_active = (target_blendshapes > 0.1).float()
    active_diff = F.mse_loss(pred_active, target_active)
    
    # 重要なブレンドシェイプにより大きな重み
    importance_weights = target_blendshapes.mean(dim=0)
    weighted_diff = F.mse_loss(
        pred_blendshapes * importance_weights,
        target_blendshapes * importance_weights
    )
    
    perceptual_loss = active_diff + weighted_diff
    total_loss += self.perceptual_weight * perceptual_loss
```

**目的**:
1. どのブレンドシェイプがアクティブかを正しく予測
2. よく使われるブレンドシェイプを重視

#### 3.4 時間的一貫性損失（オプション）
**実装場所**: `src/model/losses.py:149-165`

```python
if self.temporal_weight > 0 and pred_blendshapes.dim() == 3:
    # フレーム間の差分を計算
    pred_diff = pred_blendshapes[:, 1:, :] - pred_blendshapes[:, :-1, :]
    target_diff = target_blendshapes[:, 1:, :] - target_blendshapes[:, :-1, :]
    
    temporal_loss = F.mse_loss(pred_diff, target_diff)
    total_loss += self.temporal_weight * temporal_loss
```

**注**: SimplifiedModelでは単一フレーム出力のため使用されない

#### 3.5 ランドマーク損失（高度な機能）
**実装場所**: `src/model/losses.py:168-173`

```python
if self.use_landmark_loss:
    landmark_loss = self.landmark_loss(pred_blendshapes, target_blendshapes)
    total_loss += self.landmark_weight * landmark_loss
```

**目的**: ブレンドシェイプから顔ランドマーク位置を計算し、より直接的な制約を追加

### 4. 総合損失の計算

**最終的な損失**:
```python
total_loss = 1.0 * mse + 0.1 * l1 + 0.5 * perceptual_loss
```

**例（典型的な値）**:
- MSE: 0.02
- L1: 0.05
- Perceptual: 0.03
- **Total**: 1.0×0.02 + 0.1×0.05 + 0.5×0.03 = 0.04

### 5. 損失の勾配計算への影響

各損失成分の勾配：

#### MSE勾配
```
∂MSE/∂pred = 2 × (pred - target) / N
```

#### L1勾配
```
∂L1/∂pred = sign(pred - target) / N
```

**組み合わせ効果**:
- MSE: 滑らかな最適化
- L1: スパース性の促進
- Perceptual: 意味的な正確性

## デバッグと可視化

### 損失値の監視
```python
# 各成分を個別に記録
losses = {
    "mse": mse.item(),
    "l1": l1.item(),
    "perceptual": perceptual_loss.item(),
    "total": total_loss.item()
}
```

### 正常な損失値の範囲
- 学習初期: 0.1 - 0.5
- 収束時: 0.01 - 0.05
- 異常値: > 1.0（発散の兆候）

## 損失関数の選択理由

1. **MSE**: 基本的な再構成誤差、安定した最適化
2. **L1**: ノイズに強く、0に近い値を完全に0にする効果
3. **Perceptual**: 視覚的に重要な違いを捉える

## まとめ

このステップでは：
1. 予測と正解の差を複数の観点から計算
2. MSE（二乗誤差）、L1（絶対誤差）、知覚的損失を組み合わせ
3. 各損失に重みを付けて総合損失を算出
4. この損失値が次のバックプロパゲーションで使用される

次のステップでは、この損失からどのように勾配が計算され、モデルに伝播されるかを見ていきます。