# ステップ6: バックプロパゲーション (Backpropagation)

## 概要
損失値から各パラメータの勾配を計算し、モデルの更新方向を決定します。

## バックプロパゲーションの流れ

### 1. 勾配計算の開始
**実装場所**: `src/train.py:188`

```python
loss.backward()  # 自動微分により勾配を計算
```

**前提条件**:
- loss: スカラー値 (例: 0.04)
- requires_grad=True のパラメータが計算グラフに含まれる

### 2. 計算グラフの逆向き伝播

#### 2.1 損失から出力層への勾配
**最初の勾配**: `∂loss/∂pred_blendshapes`

```python
# MSE成分の勾配
grad_mse = 2 * (pred_blendshapes - target_blendshapes) / batch_size

# L1成分の勾配  
grad_l1 = 0.1 * torch.sign(pred_blendshapes - target_blendshapes) / batch_size

# 総合勾配
grad_output = 1.0 * grad_mse + 0.1 * grad_l1 + 0.5 * grad_perceptual
# Shape: [16, 52]
```

#### 2.2 デコーダ層の勾配伝播
**実装箇所**: デコーダのnn.Sequential内で自動処理

```python
# Sigmoid層
∂loss/∂z = grad_output * sigmoid(z) * (1 - sigmoid(z))

# Linear層 (128 → 52)
∂loss/∂W = ∂loss/∂z @ input.T
∂loss/∂b = sum(∂loss/∂z)
∂loss/∂input = W.T @ ∂loss/∂z
```

**勾配の流れ**:
1. Sigmoid: 勾配を[0, 0.25]の範囲にスケール
2. Linear3: 重み行列とバイアスの勾配を計算
3. ReLU: 正の値のみ勾配を通す
4. Dropout: 訓練時にドロップした接続は勾配0
5. Linear2, Linear1: 同様に勾配を伝播

#### 2.3 アテンション層の勾配
**最も複雑な部分**

```python
# アテンション出力の勾配から
∂loss/∂attn_output → ∂loss/∂V, ∂loss/∂attention_weights

# アテンション重みの勾配から
∂loss/∂attention_weights → ∂loss/∂scores

# スコアの勾配から
∂loss/∂scores → ∂loss/∂Q, ∂loss/∂K

# 最終的に
∂loss/∂queries  # ブレンドシェイプクエリの勾配
∂loss/∂audio_encoded  # 音声エンコーダ出力の勾配
```

**重要なポイント**:
- Softmaxの勾配は他の要素にも依存（ヤコビアン）
- マルチヘッドの場合、各ヘッドの勾配を合算

#### 2.4 音声エンコーダの勾配
```python
# audio_encodedの勾配から逆算
∂loss/∂mel_features = encoder.backward(∂loss/∂audio_encoded)
```

**エンコーダ内の勾配**:
- Linear層: 重み行列の転置を使用
- ReLU: x > 0 の場合のみ勾配を通す
- Dropout: 確率的にマスク

### 3. パラメータごとの勾配

#### 主要パラメータと典型的な勾配の大きさ

```python
# 例: 勾配の確認
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

**典型的な値**:
- `blendshape_queries.grad`: 0.001-0.01
- `decoder.6.weight.grad`: 0.01-0.1 (出力層に近い)
- `audio_encoder.0.weight.grad`: 0.0001-0.001 (入力層に近い)

### 4. 勾配の問題と対策

#### 4.1 勾配消失
**症状**: 深い層で勾配が0に近づく

**SimplifiedModelでの対策**:
- ReLU活性化（Sigmoidより勾配消失しにくい）
- 残差接続なし（浅いネットワークなので不要）

#### 4.2 勾配爆発
**症状**: 勾配が指数関数的に増大

**対策**:
```python
# 勾配クリッピング（train.pyには未実装だが推奨）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5. メモリ効率

#### 勾配の保存
- 各パラメータと同じサイズの勾配テンソルが必要
- 総メモリ: パラメータ数 × 2 × 4 bytes

#### 計算グラフの解放
```python
# backward()後、計算グラフは自動的に解放される
# 次のフォワードパスで新しいグラフが構築される
```

### 6. デバッグ用勾配チェック

```python
# 勾配が正しく流れているか確認
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: {name} has no gradient!")
        elif param.grad.norm() == 0:
            print(f"WARNING: {name} has zero gradient!")
        elif param.grad.norm() > 10:
            print(f"WARNING: {name} has large gradient: {param.grad.norm()}")
```

## 勾配フローの可視化

```
損失値 (スカラー)
    ↓ ∂loss/∂output
予測値 (16, 52)
    ↓ デコーダ.backward()
デコード前特徴 (16, 52, 256)
    ↓ アテンション.backward()
音声エンコード (16, 300, 256) + クエリ (52, 256)
    ↓ エンコーダ.backward()
メル特徴 (16, 300, 80)
    ↓ (メル特徴抽出は微分不可)
入力音声 (勾配なし)
```

## まとめ

このステップでは：
1. loss.backward()により自動微分を実行
2. 計算グラフを逆向きにたどって各パラメータの勾配を計算
3. 勾配はパラメータの.grad属性に保存される
4. 深い層ほど勾配が小さくなる傾向（勾配消失）
5. 次のステップでこれらの勾配を使ってパラメータを更新

次のステップでは、計算された勾配を使ってモデルのパラメータを更新する方法を見ていきます。