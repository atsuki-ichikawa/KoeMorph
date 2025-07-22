# ステップ7: パラメータ更新 (Parameter Update)

## 概要
計算された勾配を使ってモデルのパラメータを更新し、損失を減らす方向に学習を進めます。

## パラメータ更新の流れ

### 1. オプティマイザーによる更新
**実装場所**: `src/train.py:190`

```python
self.optimizer.step()  # 勾配を使ってパラメータを更新
```

### 2. AdamWオプティマイザーの設定
**実装場所**: `src/train.py:91-105` (setup_optimizer)

```python
self.optimizer = torch.optim.AdamW(
    self.model.parameters(),
    lr=self.config.training.learning_rate,    # 0.001
    weight_decay=self.config.training.weight_decay  # 0.01
)
```

**設定値** (configs/training/default.yaml):
```yaml
learning_rate: 0.001
weight_decay: 0.01
```

### 3. AdamWアルゴリズムの詳細

#### 3.1 運動量とRMSPropの組み合わせ

**各パラメータに対して以下の値を保持**:
```python
# 運動量 (指数移動平均)
m_t = β1 * m_{t-1} + (1 - β1) * g_t

# 二次運動量 (勾配の二乗の指数移動平均)  
v_t = β2 * v_{t-1} + (1 - β2) * g_t²
```

**デフォルト値**:
- β1 = 0.9 (運動量の減衰率)
- β2 = 0.999 (二次運動量の減衰率)
- ε = 1e-8 (数値安定性)

#### 3.2 バイアス補正
```python
# 初期ステップでの偏りを補正
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
```

#### 3.3 重み減衰（L2正則化）
```python
# パラメータに対する正則化
param = param - weight_decay * lr * param
```

#### 3.4 最終的なパラメータ更新
```python
param = param - lr * m_hat / (sqrt(v_hat) + ε)
```

### 4. 学習率スケジューリング
**実装場所**: `src/train.py:107-118` (setup_scheduler)

```python
if self.config.training.get("scheduler"):
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode="min",
        factor=0.5,        # 学習率を半分に
        patience=10,       # 10エポック改善なしで減衰
        verbose=True
    )
```

### 5. パラメータ更新の例

#### 5.1 典型的な勾配値と更新量

**ブレンドシェイプクエリの更新**:
```python
# 初期値
blendshape_queries[0] = [0.05, -0.02, 0.01, ...]  # (256次元)

# 勾配
grad = [0.001, -0.0005, 0.0002, ...]

# Adam状態更新
m = 0.9 * m_prev + 0.1 * grad  # 運動量
v = 0.999 * v_prev + 0.001 * grad²  # 二次運動量

# パラメータ更新
new_value = old_value - 0.001 * m / (sqrt(v) + 1e-8)
```

#### 5.2 レイヤー別の更新の違い

**出力層（大きな更新）**:
```python
# decoderの最終層
decoder.6.weight: grad_norm=0.05 → update_norm=0.00005
decoder.6.bias: grad_norm=0.02 → update_norm=0.00002
```

**中間層（中程度の更新）**:
```python
# アテンション層
attention.in_proj_weight: grad_norm=0.01 → update_norm=0.00001
```

**入力層（小さな更新）**:
```python
# 音声エンコーダの最初の層
audio_encoder.0.weight: grad_norm=0.002 → update_norm=0.000002
```

### 6. 勾配の初期化
**実装場所**: `src/train.py:192`

```python
self.optimizer.zero_grad()  # 次のイテレーション用に勾配をクリア
```

**重要な理由**:
- PyTorchは勾配を累積するため、明示的にクリアが必要
- バッチごとに新しい勾配を計算

### 7. パラメータ更新の監視

#### 学習率の追跡
```python
current_lr = self.optimizer.param_groups[0]['lr']
self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
```

#### 勾配ノルムの監視
```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
```

#### パラメータノルムの追跡
```python
param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
self.writer.add_scalar('train/param_norm', param_norm, self.global_step)
```

### 8. 学習の安定性

#### 適応的な学習率
- Adamは各パラメータごとに異なる学習率を自動調整
- よく更新されるパラメータ → 学習率が下がる
- あまり更新されないパラメータ → 学習率が維持される

#### 重み減衰の効果
```python
# 重み減衰 = 0.01
# 大きなパラメータほど強く正則化
penalty = 0.01 * param²
```

### 9. デバッグ用チェック

```python
def check_parameter_updates(model, old_params):
    for (name, param), (_, old_param) in zip(
        model.named_parameters(), old_params
    ):
        diff = (param - old_param).norm()
        if diff == 0:
            print(f"WARNING: {name} was not updated!")
        elif diff > 0.1:
            print(f"WARNING: Large update in {name}: {diff}")
```

### 10. 学習進行の指標

#### 良い学習の兆候
- 損失が徐々に減少
- 勾配ノルムが安定（爆発・消失なし）
- パラメータが適度に更新される

#### 問題の兆候
- 損失が振動または発散 → 学習率が高すぎる
- 勾配ノルムが0に近い → 勾配消失
- パラメータ更新が停止 → 学習率が低すぎる

## まとめ

このステップでは：
1. AdamWオプティマイザーが勾配を使ってパラメータを更新
2. 適応的学習率により各パラメータを個別に最適化
3. 重み減衰により過学習を防止
4. スケジューラーにより学習の進行に応じて学習率を調整
5. 次のイテレーション用に勾配をクリア

次のステップでは、学習の進捗を測定するためのメトリクス計算について説明します。