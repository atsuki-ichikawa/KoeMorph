# ステップ2: バッチ作成 (Batch Creation)

## 概要
DataLoaderが個別のデータサンプルを集めてバッチを作成し、GPUに転送します。

## バッチ作成の流れ

### 1. DataLoaderによるバッチ化
**実装場所**: `src/train.py:351-365` (train関数内のループ)

```python
for batch_idx, batch in enumerate(train_loader):
    # batchは16個のサンプルを含む辞書
```

**入力データ**:
- 16個の個別サンプル（DataLoaderのbatch_size=16）

**バッチの構造**:
```python
batch = {
    "wav": torch.Size([16, 160000]),    # 16個の音声データ
    "arkit": torch.Size([16, 300, 52])   # 16個のブレンドシェイプデータ
}
```

### 2. デバイスへの転送
**実装場所**: `src/train.py:173-176`

```python
# GPUへの転送
audio = batch["wav"].to(self.device)           # cuda:0へ
target_blendshapes = batch["arkit"].to(self.device)  # cuda:0へ
```

**転送後のデータ**:
- audio: `torch.cuda.FloatTensor` of size `[16, 160000]`
- target_blendshapes: `torch.cuda.FloatTensor` of size `[16, 300, 52]`

### 3. ターゲットデータの次元調整
**実装場所**: `src/train.py:178-179`

```python
if target_blendshapes.dim() == 3:
    target_blendshapes = target_blendshapes[:, 0, :]  # 最初のフレームのみ使用
```

**理由**: SimplifiedModelは単一フレームの予測を行うため

**調整後のデータ**:
- target_blendshapes: `torch.Size([16, 52])` (各サンプルの最初のフレーム)

## メモリ最適化

### Pin Memory
**効果**: CPU→GPU転送の高速化
```python
pin_memory=True  # DataLoaderの設定
```

### Drop Last
**効果**: 不完全なバッチを除外
```python
drop_last=True  # 最後の端数バッチを使わない
```

## バッチ処理の並列化

### マルチワーカー
**設定**: `num_workers=4`

**動作**:
- 4つのプロセスが並列にデータを読み込み
- メインプロセスがGPU処理を実行中に、次のバッチを準備

### データフロー
```
Worker 0 → データ読み込み → |
Worker 1 → データ読み込み → | → キューに集約 → バッチ作成 → GPU転送
Worker 2 → データ読み込み → |
Worker 3 → データ読み込み → |
```

## バッチサイズの影響

### メモリ使用量
- 音声データ: `16 × 160000 × 4 bytes = 10.24 MB`
- ブレンドシェイプ: `16 × 300 × 52 × 4 bytes = 0.998 MB`
- 合計: 約11.24 MB/バッチ

### 学習への影響
- 大きいバッチサイズ: 安定した勾配、高速な収束
- 小さいバッチサイズ: ノイズの多い勾配、正則化効果

## デバッグモード
**実装場所**: `src/train.py:191-193`

```python
if self.config.get("debug") and batch_idx > 2:
    break  # デバッグ時は3バッチのみ処理
```

## まとめ

このステップでは：
1. DataLoaderが16個のサンプルをバッチ化
2. CPUメモリからGPUメモリへ転送
3. ターゲットデータの次元を調整（時系列→単一フレーム）
4. 並列処理により次のバッチを事前準備

次のステップでは、バッチ化された音声データからメルスペクトログラム特徴を抽出します。