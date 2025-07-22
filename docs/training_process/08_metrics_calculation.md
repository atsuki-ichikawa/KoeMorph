# ステップ8: メトリクス計算 (Metrics Calculation)

## 概要
学習の進捗を評価するため、損失以外の様々な評価指標を計算します。訓練時とバリデーション時で異なるメトリクスが使用されます。

## メトリクス計算の流れ

### 1. 訓練中のメトリクス更新
**実装場所**: `src/train.py:194-195`

```python
# 訓練メトリクスを更新
self.train_metrics.update(pred_blendshapes, target_blendshapes)
```

### 2. BlendshapeMetrics クラス
**実装場所**: `src/model/losses.py:185-341`

#### 初期化
```python
class BlendshapeMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []  # 予測値のリスト
        self.targets = []      # 正解値のリスト
```

#### メトリクス蓄積
```python
def update(self, pred_blendshapes, target_blendshapes):
    """バッチごとの予測と正解を蓄積"""
    self.predictions.append(pred_blendshapes.detach().cpu())
    self.targets.append(target_blendshapes.detach().cpu())
```

### 3. 各メトリクスの詳細計算

#### 3.1 基本的な誤差指標

**MAE (Mean Absolute Error)**:
```python
mae = torch.mean(torch.abs(all_preds - all_targets))
```
- 値域: [0, 1] (ブレンドシェイプが0-1正規化されているため)
- 良い値: < 0.05
- 意味: 平均的な絶対誤差

**MSE (Mean Squared Error)**:
```python
mse = torch.mean((all_preds - all_targets) ** 2)
```
- 値域: [0, 1]
- 良い値: < 0.01
- 意味: 大きな誤差により重いペナルティ

**RMSE (Root Mean Squared Error)**:
```python
rmse = torch.sqrt(mse)
```
- 値域: [0, 1]
- 良い値: < 0.1
- 意味: MSEの平方根（元の単位と同じ）

#### 3.2 ブレンドシェイプ別統計

**ブレンドシェイプ別MAE**:
```python
bs_mae = torch.mean(torch.abs(all_preds - all_targets), dim=0)  # (52,)
max_bs_mae = torch.max(bs_mae)  # 最も誤差の大きいブレンドシェイプ
min_bs_mae = torch.min(bs_mae)  # 最も正確なブレンドシェイプ
std_bs_mae = torch.std(bs_mae)  # ブレンドシェイプ間の誤差のばらつき
```

**意味**:
- 特定のブレンドシェイプの予測が困難かを識別
- 例: 眉の動き > 唇の動き > 目の動き（一般的な傾向）

#### 3.3 相関分析

**ブレンドシェイプ別相関**:
```python
correlations = []
for i in range(all_preds.shape[1]):  # 52ブレンドシェイプ
    pred_bs = all_preds[:, i]
    target_bs = all_targets[:, i]
    
    if target_bs.std() > 1e-6:  # 分散が十分ある場合のみ
        corr = torch.corrcoef(torch.stack([pred_bs, target_bs]))[0, 1]
        correlations.append(corr)
```

**統計値**:
```python
mean_correlation = torch.mean(torch.tensor(correlations))
min_correlation = torch.min(torch.tensor(correlations))
```

**解釈**:
- 1.0: 完全な正の相関
- 0.0: 無相関
- -1.0: 完全な負の相関
- 良い値: > 0.7

#### 3.4 時間的一貫性（時系列データの場合）

**実装場所**: `src/model/losses.py:310-318`

```python
def compute_temporal_consistency(predictions, targets):
    """連続するフレーム間の変化の一貫性を測定"""
    pred_diff = predictions[:, 1:] - predictions[:, :-1]
    target_diff = targets[:, 1:] - targets[:, :-1]
    
    consistency = F.mse_loss(pred_diff, target_diff)
    return consistency
```

**注**: SimplifiedModelでは単一フレーム予測のため使用されない

#### 3.5 動作の滑らかさ

**予測の滑らかさ**:
```python
pred_smoothness = torch.mean(torch.abs(
    predictions[:, 1:] - predictions[:, :-1]
))
```

**ターゲットの滑らかさ**:
```python
target_smoothness = torch.mean(torch.abs(
    targets[:, 1:] - targets[:, :-1]
))
```

#### 3.6 活動度の測定

**予測の活動度**:
```python
pred_activity = torch.mean(torch.abs(predictions))
```

**ターゲットの活動度**:
```python
target_activity = torch.mean(torch.abs(targets))
```

**意味**: モデルが適切な動作レベルを予測しているか

#### 3.7 分類的メトリクス

**バイナリ化**:
```python
pred_binary = (all_preds > 0.1).float()   # 閾値0.1でアクティブ判定
target_binary = (all_targets > 0.1).float()
```

**精度・再現率・F1スコア**:
```python
tp = torch.sum(pred_binary * target_binary)  # True Positive
fp = torch.sum(pred_binary * (1 - target_binary))  # False Positive
fn = torch.sum((1 - pred_binary) * target_binary)  # False Negative

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1_score = 2 * precision * recall / (precision + recall + 1e-8)
```

### 4. バリデーション時の特別な処理

**実装場所**: `src/train.py:286`

```python
# バリデーション用メトリクス（より詳細）
val_metrics.update(pred_blendshapes, target_blendshapes)
```

**バリデーション専用メトリクス**:
- より詳細な統計情報
- 分布の比較
- 外れ値の検出

### 5. メトリクスの集約

**エポック終了時**:
```python
# 訓練メトリクスの計算
train_computed = self.train_metrics.compute()
self.train_metrics.reset()  # 次エポック用にリセット

# バリデーションメトリクスの計算
val_computed = self.val_metrics.compute() if validation else {}
```

### 6. メトリクスの解釈ガイド

#### 良好な学習の指標
- MAE < 0.05
- 平均相関 > 0.7
- F1スコア > 0.8
- 予測活動度 ≈ ターゲット活動度

#### 問題の診断
- 高いMAEだが高い相関 → バイアス問題（全体的にオフセット）
- 低いMAEだが低い相関 → 予測が平坦すぎる
- 低いF1スコア → 活動検出が不正確

### 7. 計算コストの最適化

#### バッチサイズの影響
```python
# メモリ使用量
predictions_memory = batch_size * num_blendshapes * 4 bytes
```

#### 効率的な実装
```python
# CPUに移動してメモリを節約
pred_detached = pred_blendshapes.detach().cpu()
```

## メトリクスの出力例

```
Epoch 25/100:
  Train - MAE: 0.042, MSE: 0.007, Mean Corr: 0.634
  Val - MAE: 0.038, MSE: 0.006, Mean Corr: 0.681, F1: 0.723
  Best Val MAE improved from 0.041 to 0.038
```

## まとめ

このステップでは：
1. 予測と正解の比較から多角的なメトリクスを計算
2. 基本誤差、相関、分類性能を総合評価
3. ブレンドシェイプ別の詳細分析
4. バリデーション時により詳細なメトリクスを計算
5. 学習の進捗と問題の診断に活用

次のステップでは、これらのメトリクスをTensorBoardに記録する方法を見ていきます。