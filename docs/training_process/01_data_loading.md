# ステップ1: 時系列データ読み込み (Sequential Data Loading)

## 概要
新しいSequentialKoeMorphDatasetは、時系列の連続性を保ちながらデータを読み込みます。
1フレームずつスライドするdense samplingから、効率的なsparse samplingまで、
学習段階に応じて柔軟に調整可能です。

## データセットの実装

### 1. SequentialKoeMorphDataset（Multi-Frame Rate対応）
**実装場所**: `src/data/sequential_dataset.py`

```python
class SequentialKoeMorphDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        window_frames: int = 256,  # 30fps: 256 (8.5s), 60fps: 512 (8.5s)
        stride_frames: int = 1,    # 1 frame stride for true sequential
        sample_rate: int = 16000,
        target_fps: int = 30,      # 30 or 60 fps
        shuffle_files: bool = True,
        loop_dataset: bool = True,
    ):
```

**重要なパラメータ（Frame Rate対応）**:
#### 30fpsモード:
- `window_frames`: 256フレーム (8.5秒) の固定ウィンドウ
- `stride_frames`: デフォルト1 (33.33msごとの密な学習)
- `hop_length`: 533サンプル (16kHz / 30fps)

#### 60fpsモード:
- `window_frames`: 512フレーム (8.5秒) の固定ウィンドウ
- `stride_frames`: デフォルト1 (16.67msごとの密な学習)
- `hop_length`: 267サンプル (16kHz / 60fps)

### 2. AdaptiveSequentialDataset
**実装場所**: `src/data/adaptive_sequential_dataset.py`

```python
class AdaptiveSequentialDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        window_frames: int = 256,
        stride_mode: str = "progressive",  # 適応的ストライド
        initial_stride: int = 32,
        final_stride: int = 1,
        epoch: int = 0,
        max_epochs: int = 100,
    ):
```

## ストライド戦略

### 1. Dense Mode (stride=1)
```python
# すべてのフレーム遷移を学習
for i in range(num_frames - window_frames + 1):
    yield window[i:i+256]  # 1フレームずつスライド
```

**利点**:
- Temporal smoothingのすべての遷移を学習
- 最高品質の時系列モデリング

**欠点**:
- 計算量が大きい (30倍のサンプル数)

### 2. Progressive Mode
```python
# エポックに応じてストライドを減少
progress = epoch / max_epochs
stride = int(initial_stride - progress * (initial_stride - final_stride))
```

**学習の流れ**:
```
エポック 0-20:  stride=32 (粗い学習、高速)
エポック 20-50: stride=16 (中間的な詳細度)
エポック 50-80: stride=8  (細かい動きの学習)
エポック 80-100: stride=1  (精密な調整)
```

### 3. Mixed Mode
```python
# 10%をdense、90%をsparse sampling
dense_samples = int(num_windows * 0.1)
dense_indices = np.random.choice(num_windows, dense_samples)
```

**サンプリング戦略**:
- 重要な遷移部分をdense sampling
- 定常部分をsparse sampling
- 効率と品質のバランス

## データ読み込みフロー

### 1. ファイルペアの検索
```python
def _find_file_pairs(self) -> List[Tuple[Path, Path]]:
    pairs = []
    for audio_path in self.data_dir.glob("**/*.wav"):
        jsonl_path = audio_path.with_suffix(".jsonl")
        if jsonl_path.exists():
            pairs.append((audio_path, jsonl_path))
    return sorted(pairs)
```

### 2. 音声データの読み込み
```python
def _load_audio(self, audio_path: Path) -> np.ndarray:
    # soundfileで高速読み込み
    audio, sr = sf.read(audio_path, dtype='float32')
    if sr != self.sample_rate:
        # リサンプリングが必要な場合
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
    return audio
```

### 3. ブレンドシェイプの読み込み（Frame Rate Resampling対応）
```python
def _load_blendshapes(self, jsonl_path: Path) -> np.ndarray:
    blendshapes = []
    timestamps = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            frame_data = json.loads(line.strip())
            blendshapes.append(frame_data['blendshapes'])
            timestamps.append(frame_data['timestamp'])
    
    blendshapes = np.array(blendshapes, dtype=np.float32)
    
    # フレームレート検出
    source_fps = 1.0 / (timestamps[1] - timestamps[0])
    
    # 必要に応じてリサンプリング
    if abs(source_fps - self.target_fps) > 1.0:
        blendshapes = self._resample_blendshapes(blendshapes, source_fps)
    
    return blendshapes

def _resample_blendshapes(self, blendshapes: np.ndarray, source_fps: float) -> np.ndarray:
    """ブレンドシェイプのフレームレート変換"""
    source_len = len(blendshapes)
    ratio = self.target_fps / source_fps
    target_len = int(source_len * ratio)
    
    # 線形補間によるリサンプリング
    source_indices = np.linspace(0, source_len - 1, target_len)
    resampled = np.zeros((target_len, blendshapes.shape[1]), dtype=np.float32)
    
    for i in range(blendshapes.shape[1]):
        resampled[:, i] = np.interp(source_indices, np.arange(source_len), blendshapes[:, i])
    
    return resampled
```

**JSONLフォーマット（Multi-Frame Rate対応）**:
```json
# 30fps データ例
{"timestamp": 0.0333, "blendshapes": [0.0, 0.2, 0.1, ...]}
{"timestamp": 0.0667, "blendshapes": [0.0, 0.25, 0.12, ...]}

# 60fps データ例  
{"timestamp": 0.0167, "blendshapes": [0.0, 0.2, 0.1, ...]}
{"timestamp": 0.0333, "blendshapes": [0.0, 0.22, 0.11, ...]}
{"timestamp": 0.0500, "blendshapes": [0.0, 0.25, 0.12, ...]}
```

### 4. ウィンドウの生成
```python
def _process_file_pair(self, audio_path: Path, jsonl_path: Path):
    audio = self._load_audio(audio_path)
    blendshapes = self._load_blendshapes(jsonl_path)
    
    # フレーム数の整合性チェック
    expected_frames = len(audio) // self.hop_length
    if abs(len(blendshapes) - expected_frames) > 1:
        # 最小値に合わせる
        num_frames = min(len(blendshapes), expected_frames)
        audio = audio[:num_frames * self.hop_length]
        blendshapes = blendshapes[:num_frames]
    
    # Sequential windows
    for i in range(0, num_frames - self.window_frames, self.stride_frames):
        start_frame = i
        end_frame = start_frame + self.window_frames
        
        start_sample = start_frame * self.hop_length
        end_sample = end_frame * self.hop_length
        
        yield {
            'audio': torch.from_numpy(audio[start_sample:end_sample]),
            'blendshapes': torch.from_numpy(blendshapes[start_frame:end_frame]),
            'file_indices': torch.tensor(file_idx),
            'start_frames': torch.tensor(start_frame),
            'file_names': audio_path.stem,
        }
```

## バッチ作成

### DataLoaderの設定
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
```

**IterableDatasetの特徴**:
- メモリ効率的（全データをメモリに載せない）
- 無限ループ可能（`loop_dataset=True`）
- マルチプロセス対応

### バッチの構造
```python
batch = {
    'audio': torch.Size([4, 136576]),      # 4サンプル × 8.5秒
    'blendshapes': torch.Size([4, 256, 52]), # 4サンプル × 256フレーム × 52値
    'file_indices': torch.Size([4]),         # ファイルID
    'start_frames': torch.Size([4]),         # 開始フレーム番号
    'file_names': ['file1', 'file1', 'file2', 'file2'],  # ファイル名
}
```

## 時系列連続性の保証

### 1. ファイル内での連続性
```python
# 同じファイルからの連続したウィンドウ
Window 1: frames[0:256]
Window 2: frames[1:257]    # 1フレームスライド
Window 3: frames[2:258]
...
```

### 2. Temporal State管理
```python
# train_sequential.pyでの状態管理
for batch in dataloader:
    file_indices = batch['file_indices']
    
    # ファイル境界の検出
    if file_indices[0] != self.current_file_idx:
        model.reset_temporal_state()  # 新しいファイル
        self.current_file_idx = file_indices[0]
```

## パフォーマンス最適化（Multi-Frame Rate対応）

### 1. メモリ使用量
#### 30fpsモード:
- **ウィンドウごと**: 8.5秒 × 16kHz × 4bytes = 544KB
- **バッチごと**: 544KB × 4 = 2.2MB (音声のみ)
- **ブレンドシェイプ**: 256フレーム × 52値 × 4bytes = 53KB/sample

#### 60fpsモード:
- **ウィンドウごと**: 8.5秒 × 16kHz × 4bytes = 544KB (音声は同じ)
- **バッチごと**: 544KB × 4 = 2.2MB (音声のみ)
- **ブレンドシェイプ**: 512フレーム × 52値 × 4bytes = 106KB/sample (2倍)

### 2. I/O最適化
```python
# soundfileによる高速読み込み
# メモリマップによる効率的アクセス
# マルチプロセスでの並列読み込み
```

### 3. キャッシング戦略
```python
# 頻繁にアクセスされるファイルはメモリに保持
# LRUキャッシュで管理（実装可能）
```

## データ拡張（オプション）

```python
# 時系列を考慮した拡張
def augment_sequential(audio, blendshapes):
    # ピッチシフト（音声とブレンドシェイプの同期維持）
    # タイムストレッチ（フレームレートの調整）
    # ノイズ追加（ロバスト性向上）
    return augmented_audio, augmented_blendshapes
```

## まとめ

新しい時系列データ読み込みシステム（Multi-Frame Rate対応）は：
1. **時系列連続性を保証**: 1フレームずつのスライディングウィンドウ（30fps/60fps対応）
2. **適応的ストライド**: 学習段階に応じた効率的なサンプリング
3. **メモリ効率**: IterableDatasetによるストリーミング処理
4. **状態管理**: ファイル境界でのtemporal state reset
5. **自動フレームレート変換**: 30fps↔60fps間の自動リサンプリング
6. **統一タイミング**: Dynamic hop_length（30fps: 533, 60fps: 267サンプル）で同期
7. **効率的リサンプリング**: 線形補間による高品質フレームレート変換

次のステップでは、これらのウィンドウからバッチを作成する方法を説明します。