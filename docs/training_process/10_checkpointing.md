# ステップ10: チェックポイント保存 (Checkpointing)

## 概要
学習中にモデルの状態を定期的に保存し、学習の中断・再開や最適なモデルの選択を可能にします。

## チェックポイント保存の流れ

### 1. 保存タイミングの決定
**実装場所**: `src/train.py:239-246`

```python
# 最良モデルかどうかの判定
is_best = False
if val_metrics and "loss" in val_metrics:
    current_val_loss = val_metrics["loss"]
    if current_val_loss < self.best_val_loss:
        self.best_val_loss = current_val_loss
        is_best = True

# チェックポイント保存
self.save_checkpoint(train_metrics, is_best)
```

### 2. チェックポイントデータの構築
**実装場所**: `src/train.py:302-313`

```python
checkpoint = {
    "epoch": self.current_epoch,
    "global_step": self.global_step,
    "model_state_dict": self.model.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "metrics": metrics,
    "config": OmegaConf.to_container(self.config),
}

if self.scheduler:
    checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
```

#### チェックポイントに含まれる情報

**学習状態**:
- `epoch`: 現在のエポック番号
- `global_step`: 総ステップ数（バッチ数の累計）

**モデル情報**:
- `model_state_dict`: モデルのパラメータ（重みとバイアス）
- `config`: モデル構成の設定

**オプティマイザ状態**:
- `optimizer_state_dict`: Adam状態（運動量、二次運動量）
- `scheduler_state_dict`: 学習率スケジューラーの状態

**評価指標**:
- `metrics`: 現在のエポックでのメトリクス

### 3. 保存ファイルの種類

#### 3.1 通常のチェックポイント
**実装場所**: `src/train.py:315-318`

```python
checkpoint_path = (
    self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch:03d}.pth"
)
torch.save(checkpoint, checkpoint_path)
```

**ファイル名例**: `checkpoint_epoch_025.pth`

#### 3.2 最良モデルの保存
**実装場所**: `src/train.py:320-324`

```python
if is_best:
    best_path = self.checkpoint_dir / "best_model.pth"
    torch.save(checkpoint, best_path)
    logger.info(f"Saved best model with val_loss: {metrics.get('loss', 0):.4f}")
```

#### 3.3 最新モデルの保存
**実装場所**: `src/train.py:326-328`

```python
last_path = self.checkpoint_dir / "last_model.pth"
torch.save(checkpoint, last_path)
```

### 4. 保存ディレクトリ構造

```
outputs/2025-07-17/19-20-14/
└── checkpoints/
    ├── best_model.pth           # バリデーション損失最小のモデル
    ├── last_model.pth           # 最新のエポック
    ├── checkpoint_epoch_000.pth # エポックごとの保存
    ├── checkpoint_epoch_001.pth
    ├── checkpoint_epoch_002.pth
    └── ...
```

### 5. チェックポイント読み込み

#### 5.1 学習再開用の読み込み
**実装場所**: `src/train.py:330-343`

```python
def _load_checkpoint(self, checkpoint_path: str):
    """学習を途中から再開する場合"""
    checkpoint = torch.load(checkpoint_path, map_location=self.device)

    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint and self.scheduler:
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    self.current_epoch = checkpoint.get("epoch", 0)
    self.global_step = checkpoint.get("global_step", 0)

    logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
```

#### 5.2 推論用の読み込み
**実装場所**: `scripts/test_model.py:45`

```python
def load_for_inference(checkpoint_path: str, config):
    """推論のみの場合（オプティマイザ状態は不要）"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = SimplifiedKoeMorphModel(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, checkpoint.get("metrics", {})
```

### 6. 保存頻度の最適化

#### 6.1 設定による制御
```yaml
# configs/training/default.yaml
checkpoint:
  save_freq: 5        # 5エポックごとに保存
  keep_last_n: 10     # 最新10個のチェックポイントを保持
  save_best_only: false  # 最良モデルのみ保存するか
```

#### 6.2 効率的な保存戦略
```python
def smart_checkpoint_saving(self, epoch, metrics):
    """重要なタイミングでのみ保存"""
    should_save = (
        epoch % self.config.training.checkpoint.save_freq == 0  # 定期保存
        or metrics.get("val_loss", float('inf')) < self.best_val_loss  # 改善時
        or epoch in [10, 25, 50, 100, 200]  # 特定のマイルストーン
    )
    
    if should_save:
        self.save_checkpoint(metrics, is_best)
```

### 7. ディスク容量の管理

#### 7.1 古いチェックポイントの削除
```python
def cleanup_old_checkpoints(self, keep_n=10):
    """最新N個以外の通常チェックポイントを削除"""
    checkpoint_files = sorted(
        self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for old_file in checkpoint_files[keep_n:]:
        old_file.unlink()  # ファイル削除
        logger.info(f"Removed old checkpoint: {old_file.name}")
```

#### 7.2 圧縮保存
```python
# 古いチェックポイントを圧縮
import gzip
import pickle

def save_compressed_checkpoint(checkpoint, path):
    with gzip.open(f"{path}.gz", 'wb') as f:
        pickle.dump(checkpoint, f)
```

### 8. チェックポイントの検証

#### 8.1 整合性チェック
```python
def validate_checkpoint(checkpoint_path):
    """チェックポイントが正しく保存されているか確認"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        required_keys = ["model_state_dict", "epoch", "config"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            return False
            
        # モデル状態の基本チェック
        state_dict = checkpoint["model_state_dict"]
        if not state_dict or len(state_dict) == 0:
            logger.warning("Empty model state dict")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return False
```

#### 8.2 メタデータの保存
```python
# チェックポイントと一緒にメタデータを保存
metadata = {
    "timestamp": time.time(),
    "hostname": socket.gethostname(),
    "git_commit": get_git_commit(),
    "pytorch_version": torch.__version__,
    "model_params": sum(p.numel() for p in model.parameters()),
    "training_time": training_elapsed_time
}

checkpoint["metadata"] = metadata
```

### 9. 異なる形式での保存

#### 9.1 ONNX形式での保存
```python
def export_onnx(model, sample_input, output_path):
    """推論に最適化されたONNX形式で保存"""
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=["audio"],
        output_names=["blendshapes"],
        dynamic_axes={"audio": {0: "batch_size"}}
    )
```

#### 9.2 TorchScript形式での保存
```python
def export_torchscript(model, output_path):
    """モバイル対応のTorchScript形式で保存"""
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
```

### 10. チェックポイント利用の例

#### 学習再開
```bash
python src/train.py --resume outputs/2025-07-17/19-20-14/checkpoints/last_model.pth
```

#### ベストモデルでの評価
```bash
python scripts/test_model.py --checkpoint outputs/2025-07-17/19-20-14/checkpoints/best_model.pth
```

#### 特定エポックからの再開
```bash
python src/train.py --resume outputs/2025-07-17/19-20-14/checkpoints/checkpoint_epoch_050.pth
```

### 11. デバッグ用機能

#### チェックポイント情報の表示
```python
def inspect_checkpoint(checkpoint_path):
    """チェックポイントの詳細情報を表示"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Global step: {checkpoint['global_step']}")
    print(f"Metrics: {checkpoint.get('metrics', {})}")
    print(f"Model parameters: {len(checkpoint['model_state_dict'])}")
    
    # パラメータサイズの表示
    total_params = sum(
        p.numel() for p in checkpoint['model_state_dict'].values()
    )
    print(f"Total parameters: {total_params:,}")
```

## まとめ

このステップでは：
1. 学習の重要な時点でモデル状態を自動保存
2. 最良モデル、最新モデル、定期チェックポイントの管理
3. 学習再開に必要な全情報の保存
4. ディスク容量を考慮した効率的な保存戦略
5. チェックポイントの整合性検証とメタデータ管理

これで SimplifiedKoeMorphModel の学習過程全10ステップの詳細解説が完了しました。各ステップが連携して効果的な学習を実現しています。