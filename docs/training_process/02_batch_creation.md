# ステップ2: 時系列バッチ作成 (Sequential Batch Creation)

## 概要
AdaptiveSequentialDatasetから時系列順のウィンドウデータを受け取り、
時系列の連続性を保ちながらバッチを作成します。

## 時系列バッチ作成の流れ

### 1. Sequential DataLoaderによるバッチ化
**実装場所**: `src/train_sequential.py:SequentialTrainer.train_epoch()`

```python
for batch_idx, batch in enumerate(self.train_loader):
    # batchは4個の連続する時系列ウィンドウを含む
    file_indices = batch['file_indices']
    start_frames = batch['start_frames']
```

**入力データ**:
- 4個の時系列ウィンドウ（DataLoaderのbatch_size=4）
- 各ウィンドウは256フレーム（8.5秒）

**バッチの構造**:
```python
batch = {
    'audio': torch.Size([4, 136576]),      # 4ウィンドウ × 8.5秒音声
    'blendshapes': torch.Size([4, 256, 52]), # 4ウィンドウ × 256フレーム × 52値
    'file_indices': torch.Size([4]),         # ファイルID
    'start_frames': torch.Size([4]),         # 開始フレーム番号
    'file_names': ['file1', 'file1', 'file2', 'file2'],  # ファイル名
    'is_dense': torch.Size([4]),             # Dense samplingフラグ
}
```

### 2. Temporal State管理
**実装場所**: `src/train_sequential.py:SequentialTrainer.train_epoch()`

```python
# ファイル境界の検出と状態リセット
if file_idx != self.current_file_idx:
    self.model.reset_temporal_state()  # 新しいファイル
    self.current_file_idx = file_idx
    logger.info(f"Reset temporal state for file {file_idx}")
```

**重要性**:
- ファイル間での時系列の連続性を切断
- Temporal smoothingの前フレーム情報をクリア
- 異なる話者/セッション間の干渉を防止

### 3. デバイスへの転送
```python
# GPUへの転送
audio = batch['audio'].to(self.device)           # cuda:0へ
target_blendshapes = batch['blendshapes'].to(self.device)  # cuda:0へ
```

**転送後のデータ**:
- audio: `torch.cuda.FloatTensor` of size `[4, 136576]`
- target_blendshapes: `torch.cuda.FloatTensor` of size `[4, 256, 52]`

### 4. ウィンドウからフレーム抽出
**実装場所**: 時系列モデルでの処理

```python
# 256フレームウィンドウから最後のフレームを予測
# （前の255フレームを文脈として使用）
pred_blendshapes = model(audio)  # (B, 52) - 最終フレームのみ予測
target_frame = target_blendshapes[:, -1, :]  # (B, 52) - 最後のフレーム
```

**理由**: 時系列文脈を使って次フレームを予測する因果的な学習

## 適応的ストライド処理

### Progressive Stride Mode
**実装場所**: `src/data/adaptive_sequential_dataset.py`

```python
# エポックに応じてストライドを調整
progress = min(1.0, epoch / max(1, max_epochs - 1))
stride = int(initial_stride - progress * (initial_stride - final_stride))

# 学習の進行例
エポック 0-20:  stride=32 (粗い学習、高速)
エポック 20-50: stride=16 (中間的な詳細度)
エポック 50-80: stride=8  (細かい動きの学習)
エポック 80-100: stride=1  (精密な調整)
```

### Mixed Mode
```python
# 10%をdense、90%をsparse sampling
dense_samples = int(num_windows * 0.1)
dense_indices = np.random.choice(num_windows, dense_samples)
```

## メモリ最適化

### Pin Memory
**効果**: CPU→GPU転送の高速化
```python
pin_memory=torch.cuda.is_available()  # DataLoaderの設定
```

### Drop Last
**効果**: 不完全なバッチを除外
```python
drop_last=True  # 最後の端数バッチを使わない
```

### Persistent Workers
```python
persistent_workers=num_workers > 0  # ワーカープロセスを維持
```

## バッチ処理の並列化

### マルチワーカー
**設定**: `num_workers=2` (時系列処理では少なめ)

**IterableDataset対応**:
- 各ワーカーが独立してファイルを処理
- 時系列順序を保持しながら並列化
- メモリ効率的なストリーミング処理

### データフロー
```
Worker 0 → File A sequential windows → |
Worker 1 → File B sequential windows → | → バッチ集約 → GPU転送
                                       |
                    Temporal State管理 →|
```

## バッチサイズの影響

### メモリ使用量
- 音声データ: `4 × 136576 × 4 bytes = 2.18 MB`
- ブレンドシェイプ: `4 × 256 × 52 × 4 bytes = 0.21 MB`
- 合計: 約2.4 MB/バッチ

**時系列処理での特徴**:
- より長い音声コンテキスト（8.5秒 vs 10秒）
- 小さいバッチサイズ（時系列の複雑性のため）
- 効率的なメモリ使用

### 時系列学習への影響
- **Dense sampling**: 高品質だが計算量大
- **Progressive sampling**: 効率的な段階的学習
- **Mixed sampling**: バランスの取れたアプローチ

## デバッグモード
**実装場所**: `src/train_sequential.py`

```python
if self.debug and batch_idx > 2:
    break  # デバッグ時は3バッチのみ処理
```

## ファイル境界の処理

### Temporal State Reset
```python
# 新しいファイルの開始を検出
if file_idx != self.current_file_idx:
    self.model.reset_temporal_state()
    self.current_file_idx = file_idx
```

**重要性**:
- 異なるセッション間での状態混在を防止
- Temporal smoothingの適切な初期化
- ファイル単位での独立した処理

## まとめ

時系列バッチ作成では：
1. **Sequential Dataset**が時系列順にウィンドウを生成
2. **Adaptive Stride**で効率的な学習密度を調整
3. **Temporal State管理**でファイル境界を適切に処理
4. **メモリ効率**を保ちながら長い音声コンテキストを処理
5. **因果的学習**で次フレーム予測を実行

次のステップでは、バッチ化された音声データからデュアルストリーム特徴（Mel + Emotion）を並列抽出します。