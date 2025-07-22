# ステップ3: 音声特徴抽出 (Audio Feature Extraction)

## 概要
SimplifiedKoeMorphModelは生の音声波形からメルスペクトログラムを抽出し、モデルの入力として使用します。

## 特徴抽出の流れ

### 1. extract_mel_features関数の呼び出し
**実装場所**: `src/model/simplified_model.py:127` (forward関数内)

```python
mel_features = self.extract_mel_features(audio)  # (B, T, n_mels)
```

**入力**:
- audio: `torch.Size([16, 160000])` (16個の10秒音声 @ 16kHz)

### 2. バッチ処理とCPU転送
**実装場所**: `src/model/simplified_model.py:79-87`

```python
batch_size, audio_length = audio.shape
mel_features = []
for i in range(batch_size):
    audio_np = audio[i].cpu().numpy()  # GPUからCPUへ転送
```

**理由**: LibrosaはNumPy配列で動作するため

### 3. メルスペクトログラム計算
**実装場所**: `src/model/simplified_model.py:89-97`

```python
mel = librosa.feature.melspectrogram(
    y=audio_np,
    sr=self.sample_rate,      # 16000 Hz
    n_fft=self.n_fft,         # 1024
    hop_length=self.hop_length, # 533 (≈16000/30)
    n_mels=self.n_mels,       # 80
    fmin=80,                  # 最小周波数 80Hz
    fmax=8000                 # 最大周波数 8000Hz
)
```

**パラメータの意味**:
- `n_fft=1024`: FFTウィンドウサイズ（64ms @ 16kHz）
- `hop_length=533`: フレームシフト（33.3ms、30fps相当）
- `n_mels=80`: メルフィルタバンクの数
- `fmin/fmax`: 人間の音声に最適化された周波数範囲

**出力形状**:
- mel: `(80, 300)` (80メル周波数ビン × 300時間フレーム)

### 4. デシベル変換と正規化
**実装場所**: `src/model/simplified_model.py:100-101`

```python
mel = librosa.power_to_db(mel, ref=np.max)  # パワーをdBに変換
mel = (mel + 80) / 80  # [-80, 0] → [0, 1]に正規化
```

**変換の詳細**:
1. パワースペクトログラム → デシベルスケール
2. 最大値を基準（0dB）として相対値に変換
3. 通常の範囲（-80dB〜0dB）を0〜1に正規化

### 5. 転置とバッチ集約
**実装場所**: `src/model/simplified_model.py:103-111`

```python
mel_features.append(mel.T)  # (T, n_mels)に転置

# パディングして長さを揃える
max_length = max(feat.shape[0] for feat in mel_features)
padded_features = np.zeros((batch_size, max_length, self.n_mels))
```

**パディング処理**:
- 各音声の長さが微妙に異なる場合に対応
- ゼロパディングで最大長に揃える

### 6. GPUへの転送
**実装場所**: `src/model/simplified_model.py:112`

```python
return torch.tensor(padded_features, dtype=torch.float32, device=audio.device)
```

**最終出力**:
- mel_features: `torch.Size([16, 300, 80])` 
- デバイス: cuda:0
- 値域: [0, 1]

## 特徴抽出の可視化

メルスペクトログラムの例（1サンプル）:
```
周波数(メル) ↑
8000Hz |████░░░░░░░░░░░░░░░░|  高周波成分
       |████████░░░░░░░░░░░░|
       |████████████░░░░░░░░|  フォルマント
       |████████████████░░░░|
  80Hz |████████████████████|  基本周波数
       +--------------------→ 時間 (10秒)
```

## 計算コスト

### 時間計算量
- FFT: O(n_fft × log(n_fft)) × フレーム数
- 1バッチ（16サンプル）: 約50-100ms

### メモリ使用量
- 入力: 16 × 160000 × 4 bytes = 10.24 MB
- 出力: 16 × 300 × 80 × 4 bytes = 1.54 MB

## なぜメルスペクトログラムか？

1. **人間の聴覚特性を反映**: メル尺度は人間の周波数知覚に対応
2. **音声の特徴を効率的に表現**: 音素、イントネーション、感情を含む
3. **次元削減**: 160000サンプル → 300×80の特徴マップ
4. **時間-周波数の局在性**: 音声の時間変化と周波数成分を同時に捉える

## まとめ

このステップでは：
1. バッチ内の各音声をCPUに転送してLibrosaで処理
2. メルスペクトログラムを計算（80メル × 300フレーム）
3. デシベル変換と正規化で[0,1]の範囲に
4. バッチ全体をパディングして形状を統一
5. GPUに転送して次の処理へ

次のステップでは、これらのメル特徴がモデル内部でどのように処理されるかを見ていきます。