# ステップ1: データ読み込み (Data Loading)

## 概要
このステップでは、ディスクから学習データを読み込みます。KoeMorphDatasetクラスが各データファイルのペア（音声ファイルとブレンドシェイプファイル）を管理します。

## データの保存場所

```
/home/ichikawa/KoeMorph/output/organized/train/
├── train_00000.wav      # 音声ファイル (16kHz, モノラル)
├── train_00000.jsonl    # ARKitブレンドシェイプファイル
├── train_00001.wav
├── train_00001.jsonl
└── ... (合計114ファイルペア)
```

## データ読み込みの流れ

### 1. データセット初期化
**実装場所**: `src/data/dataset.py:45-80` (KoeMorphDataset.__init__)

```python
self.data_dir = Path("/home/ichikawa/KoeMorph/output/organized/train")
self.sample_rate = 16000
self.target_fps = 30
self.max_audio_length = 10.0  # 最大10秒
```

**処理内容**:
- ディレクトリ内の全JSONLファイルを検索
- 対応するWAVファイルの存在確認
- ファイルペアのリストを作成

**データ状態**:
```python
self.file_pairs = [
    ("train_00000.jsonl", "train_00000.wav"),
    ("train_00001.jsonl", "train_00001.wav"),
    ...
]
```

### 2. 個別データ読み込み (__getitem__)
**実装場所**: `src/data/dataset.py:82-178` (KoeMorphDataset.__getitem__)

#### 2.1 音声ファイル読み込み
**実装場所**: `src/data/io.py:40-72` (load_audio)

**入力データ**:
- ファイルパス: `/home/ichikawa/KoeMorph/output/organized/train/train_00000.wav`

**処理**:
```python
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
```

**出力データ**:
- 形状: `(audio_length,)` (例: `(160000,)` for 10秒)
- 型: `numpy.ndarray (float32)`
- 値域: `[-1.0, 1.0]`

#### 2.2 ブレンドシェイプファイル読み込み
**実装場所**: `src/data/io.py:75-113` (load_arkit_blendshapes)

**入力データ**:
- ファイルパス: `/home/ichikawa/KoeMorph/output/organized/train/train_00000.jsonl`

**JSONLファイルの中身**:
```json
{"timestamp": 0.033, "blendshapes": [0.0, 0.2, 0.8, ...]}  // 52個の値
{"timestamp": 0.066, "blendshapes": [0.1, 0.3, 0.7, ...]}
...
```

**処理**:
1. 各行をパースしてタイムスタンプとブレンドシェイプ値を取得
2. NumPy配列に変換

**出力データ**:
- timestamps: `(num_frames,)` (例: `(300,)` for 10秒@30fps)
- blendshapes: `(num_frames, 52)`
- 型: `numpy.ndarray (float32)`
- 値域: `[0.0, 1.0]`

### 3. データ同期とパディング
**実装場所**: `src/data/dataset.py:147-170`

**処理内容**:
1. 音声とブレンドシェイプの長さを同期
2. 最大長に満たない場合はパディング
3. 最大長を超える場合はトリミング

**最終的なデータ形状**:
- audio: `(160000,)` (10秒 @ 16kHz)
- blendshapes: `(300, 52)` (10秒 @ 30fps)

### 4. テンソル変換
**実装場所**: `src/data/dataset.py:172-178`

```python
return {
    "wav": torch.from_numpy(audio).float(),        # torch.Size([160000])
    "arkit": torch.from_numpy(blendshapes).float() # torch.Size([300, 52])
}
```

## DataLoaderによるバッチ化

**実装場所**: `src/data/dataset.py:265-275` (train_dataloader)

```python
DataLoader(
    dataset=self._train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
```

**バッチデータの形状**:
- `batch["wav"]`: `torch.Size([16, 160000])`
- `batch["arkit"]`: `torch.Size([16, 300, 52])`

## まとめ

このステップでは：
1. ディスクから音声ファイル（WAV）とブレンドシェイプファイル（JSONL）を読み込み
2. 音声を16kHz、ブレンドシェイプを30fpsに統一
3. 長さを10秒に揃えてパディング/トリミング
4. PyTorchテンソルに変換してバッチ化

次のステップでは、これらのバッチデータがどのように処理されるかを見ていきます。