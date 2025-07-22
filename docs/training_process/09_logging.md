# ステップ9: ログ記録 (Logging)

## 概要
学習の進捗と各種メトリクスをTensorBoardとコンソール出力に記録し、学習過程を可視化・監視します。

## ログ記録の流れ

### 1. TensorBoard Writer の初期化
**実装場所**: `src/train.py:60-61`

```python
self.log_dir = self.output_dir / "logs"
self.writer = SummaryWriter(str(self.log_dir))
```

**ログディレクトリ**:
```
outputs/2025-07-17/19-20-14/
└── logs/
    └── events.out.tfevents.1752747615.aitnlplab1.94184.0
```

### 2. 訓練中のリアルタイムログ

#### 2.1 バッチレベルのログ記録
**実装場所**: `src/train.py:196-209`

```python
# ステップごとの損失を記録
self.writer.add_scalar("train/loss", loss.item(), self.global_step)

# 学習率の記録
current_lr = self.optimizer.param_groups[0]["lr"]
self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

# 勾配ノルムの記録（オプション）
if self.config.training.get("log_grad_norm"):
    total_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), float('inf')
    )
    self.writer.add_scalar("train/grad_norm", total_norm, self.global_step)
```

#### 2.2 進捗バーの表示
**実装場所**: `src/train.py:175`

```python
with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.training.max_epochs}") as pbar:
    for batch_idx, batch in enumerate(pbar):
        # ... 学習処理 ...
        
        # 進捗バーの説明を更新
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
```

**出力例**:
```
Epoch 25/100: 100%|████████| 100/100 [02:15<00:00, loss=0.0234, lr=1.00e-03]
```

### 3. エポック終了時のログ

#### 3.1 訓練メトリクスの記録
**実装場所**: `src/train.py:214-219`

```python
# 訓練メトリクスの計算と記録
train_metrics = self.train_metrics.compute()
for metric_name, value in train_metrics.items():
    self.writer.add_scalar(f"train/{metric_name}", value, epoch)
    
self.train_metrics.reset()
```

#### 3.2 バリデーションメトリクスの記録
**実装場所**: `src/train.py:222-227`

```python
if epoch % self.config.training.validation_freq == 0:
    val_metrics = self.validate()
    
    for metric_name, value in val_metrics.items():
        self.writer.add_scalar(f"val/{metric_name}", value, epoch)
```

#### 3.3 コンソール出力
**実装場所**: `src/train.py:229-237`

```python
# 詳細なログ出力
logger.info(f"Epoch {epoch+1:3d}/{self.config.training.max_epochs}")
logger.info("-" * 50)

# 訓練メトリクス
logger.info("Train Metrics:")
for metric_name, value in train_metrics.items():
    logger.info(f"  {metric_name}: {value:.4f}")

# バリデーションメトリクス
if val_metrics:
    logger.info("Validation Metrics:")
    for metric_name, value in val_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
```

**出力例**:
```
INFO - Epoch  25/100
INFO - --------------------------------------------------
INFO - Train Metrics:
INFO -   mae: 0.0421
INFO -   mse: 0.0072
INFO -   mean_correlation: 0.6341
INFO - Validation Metrics:
INFO -   mae: 0.0387
INFO -   mse: 0.0063
INFO -   mean_correlation: 0.6812
```

### 4. 高度な可視化

#### 4.1 分布のヒストグラム
```python
# ブレンドシェイプの分布を記録
if epoch % 10 == 0:  # 10エポックごと
    with torch.no_grad():
        sample_pred = pred_blendshapes[0]  # 最初のサンプル
        sample_target = target_blendshapes[0]
        
        self.writer.add_histogram("pred_distribution", sample_pred, epoch)
        self.writer.add_histogram("target_distribution", sample_target, epoch)
```

#### 4.2 画像として保存
```python
# 予測と正解の比較プロット
if epoch % 50 == 0:
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = range(52)  # ブレンドシェイプ番号
    ax.bar(x, sample_pred.cpu(), alpha=0.7, label="Predicted")
    ax.bar(x, sample_target.cpu(), alpha=0.7, label="Target")
    
    ax.set_xlabel("Blendshape Index")
    ax.set_ylabel("Value")
    ax.set_title(f"Epoch {epoch}: Predicted vs Target")
    ax.legend()
    
    self.writer.add_figure("blendshape_comparison", fig, epoch)
    plt.close(fig)
```

#### 4.3 音声波形の可視化
```python
# 入力音声の可視化
self.writer.add_audio("input_audio", audio[0], epoch, sample_rate=16000)

# メルスペクトログラムの可視化
mel_img = mel_features[0].transpose(0, 1).unsqueeze(0)  # (1, 80, 300)
self.writer.add_image("mel_spectrogram", mel_img, epoch)
```

### 5. ファイルベースのログ

#### 5.1 テキストログファイル
**実装場所**: `src/train.py:28-30`

```python
# ファイルハンドラーの追加
log_file = self.output_dir / "train.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)
```

**ログファイルの内容**:
```
2024-07-17 19:20:14,123 - INFO - Starting training...
2024-07-17 19:20:15,456 - INFO - Epoch   1/100
2024-07-17 19:20:15,457 - INFO - Train loss: 0.1234
2024-07-17 19:22:30,789 - INFO - Validation loss: 0.1123
```

#### 5.2 CSV形式での記録
```python
# メトリクスの履歴をCSV保存（カスタム実装）
import csv

def log_metrics_to_csv(self, epoch, train_metrics, val_metrics):
    csv_file = self.output_dir / "metrics.csv"
    
    # ヘッダーの書き込み（初回のみ）
    if epoch == 0:
        headers = ["epoch"] + list(train_metrics.keys()) + [f"val_{k}" for k in val_metrics.keys()]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    # データの書き込み
    row = [epoch] + list(train_metrics.values()) + list(val_metrics.values())
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
```

### 6. デバッグ用ログ

#### 6.1 詳細なデバッグ情報
```python
if self.config.get("debug"):
    logger.debug(f"Batch {batch_idx}: input shape = {audio.shape}")
    logger.debug(f"Mel features shape = {mel_features.shape}")
    logger.debug(f"Model output shape = {pred_blendshapes.shape}")
    
    # NaNやInfの検出
    if torch.isnan(loss):
        logger.error(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
        break
```

#### 6.2 モデルの重みの監視
```python
# パラメータの統計を記録
for name, param in self.model.named_parameters():
    if param.requires_grad:
        self.writer.add_histogram(f"weights/{name}", param, epoch)
        
        if param.grad is not None:
            self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### 7. ログの最適化

#### 7.1 ログ頻度の調整
```python
# 設定ファイルでログ頻度を制御
if self.global_step % self.config.training.log_freq == 0:
    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
```

#### 7.2 メモリ効率の考慮
```python
# 大きなデータのログは間隔を空けて
if epoch % 100 == 0:  # 100エポックごとのみ
    self.writer.add_audio("sample_audio", audio[0], epoch)
```

### 8. TensorBoardの起動と確認

```bash
# TensorBoardサーバーの起動
tensorboard --logdir=outputs/2025-07-17/19-20-14/logs --host=0.0.0.0 --port=6006
```

**確認できる項目**:
- スカラー値（損失、メトリクス、学習率）
- ヒストグラム（パラメータ、勾配、予測分布）
- 画像（メルスペクトログラム、比較プロット）
- 音声（入力サンプル）

### 9. ログ分析の例

#### 学習曲線の確認
```python
# TensorBoardで以下を確認
- train/loss vs val/loss → 過学習の検出
- train/mae vs val/mae → 汎化性能の確認
- learning_rate → 学習率スケジューリングの効果
```

#### 異常検出
```python
- 損失の急激な増加 → 勾配爆発
- メトリクスの停滞 → 学習の停止
- NaN値の出現 → 数値不安定性
```

## まとめ

このステップでは：
1. TensorBoardによる多角的な学習過程の可視化
2. コンソールとファイルへの詳細ログ出力
3. 進捗バーによるリアルタイム監視
4. デバッグ用の詳細情報記録
5. 学習曲線や分布の可視化による異常検出

次のステップでは、学習過程で重要なチェックポイント保存について説明します。