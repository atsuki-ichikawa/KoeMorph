# Dual-Stream Sequential Training Process

このドキュメントシリーズでは、KoeMorphのデュアルストリーム時系列学習プロセスを詳細に解説します。
新しいアーキテクチャでは、mel-spectrogramとemotion特徴を独立したストリームで処理し、
時系列の連続性を保ちながら学習を行います。

## 学習プロセスの概要

KoeMorphのデュアルストリーム学習は以下のステップで構成されています：

1. **[時系列データ読み込み](01_data_loading.md)** - Sequential datasetによる時系列順データ読み込み
2. **[適応的ウィンドウ生成](02_batch_creation.md)** - 1フレーム～128フレームの適応的ストライド
3. **[デュアルストリーム特徴抽出](03_audio_feature_extraction.md)** - Mel (8.5s) + Emotion (20s) の並列抽出
4. **[デュアルストリームフォワードパス](04_model_forward_pass.md)** - 自然な特化を学習する並列処理
5. **[時系列考慮型損失計算](05_loss_calculation.md)** - Temporal smoothingを含む損失
6. **[バックプロパゲーション](06_backpropagation.md)** - 時系列状態を保持した勾配計算
7. **[パラメータ更新](07_parameter_update.md)** - AdamWとCosine Annealing
8. **[シーケンス単位メトリクス](08_metrics_calculation.md)** - ファイル単位での評価
9. **[ログ記録](09_logging.md)** - TensorBoardへの詳細記録
10. **[チェックポイント保存](10_checkpointing.md)** - 時系列状態を含むモデル保存

## アーキテクチャの主要変更点

### 1. デュアルストリーム処理
```
Mel-spectrogram (8.5s context, 30fps: 33.3ms / 60fps: 16.7ms updates)
    ↓
Mel Stream → 口の動き特化
                      ↘
                       Dual-Stream Cross-Attention → Blendshapes
                      ↗
Emotion Stream → 表情全体特化
    ↑
Emotion Features (20s context, 300ms updates)
```

### 2. 時系列連続性の保持
- **Temporal Smoothing**: 前フレーム情報を使った平滑化
- **Sequential Windows**: 1フレームずつスライドする密な学習
- **File-wise Processing**: ファイル境界でのみ状態リセット
- **Multi-Frame Rate**: 30fps/60fps対応と自動リサンプリング

### 3. 統一されたタイミング（Multi-Frame Rate対応）
#### 30fps モード:
- **hop_length**: `int(16000 / 30) = 533` サンプル
- **Frame interval**: 33.33ms
- **Mel updates**: 毎フレーム (30fps)
- **Emotion updates**: ~9フレームごと (300ms)

#### 60fps モード:
- **hop_length**: `int(16000 / 60) = 267` サンプル  
- **Frame interval**: 16.67ms
- **Mel updates**: 毎フレーム (60fps)
- **Emotion updates**: ~18フレームごと (300ms)

## ファイル構成

```
KoeMorph/
├── src/
│   ├── train_sequential.py         # 時系列学習スクリプト（30fps/60fps対応）
│   ├── model/
│   │   ├── simplified_dual_stream_model.py   # デュアルストリームモデル（基本）
│   │   ├── sequential_dual_stream_model.py   # 時系列出力モデル（推奨）
│   │   └── dual_stream_attention.py          # 自然特化クロスアテンション
│   ├── data/
│   │   ├── sequential_dataset.py             # 時系列データセット（フレームレート対応）
│   │   └── adaptive_sequential_dataset.py    # 適応的ストライド
│   └── features/
│       ├── mel_sliding_window.py             # Melスライディングウィンドウ
│       ├── opensmile_extractor.py            # OpenSMILE eGeMAPS（3窓連結）
│       └── emotion_extractor.py              # 統合emotion抽出
└── configs/
    ├── model/dual_stream.yaml                # モデル設定
    ├── dual_stream_config.yaml               # 学習設定（30fps）
    └── experiment/dual_stream_60fps.yaml     # 60fps実験設定
```

## データ形式

### 入力
- **音声**: 16kHz, モノラル WAVファイル
- **Blendshapes**: ARKit 52値 (JSONLファイル, 30fps または 60fps)

### 特徴量（Multi-Frame Rate対応）
#### 30fpsモード:
- **Mel**: 80次元, 256フレーム (8.5秒), 33.3ms更新
- **Emotion**: 256次元 (Enhanced 3窓連結), 20秒コンテキスト, 300ms更新

#### 60fpsモード:
- **Mel**: 80次元, 512フレーム (8.5秒), 16.7ms更新  
- **Emotion**: 256次元 (Enhanced 3窓連結), 20秒コンテキスト, 300ms更新

### 出力
- **Blendshapes**: 52次元 [0,1], 30fps または 60fps（設定可能）
- **Sequential Output**: 完全な時系列生成（単一フレームから時系列へ拡張）
- **Temporal consistency**: 前フレームとの連続性保持
- **Automatic Resampling**: 異なるフレームレート間の自動変換

## 学習戦略

### Progressive Stride Mode
```python
# エポックとともにストライドを減少
初期エポック: stride=32 (高速・粗い学習)
    ↓
中期エポック: stride=8 (中程度の詳細)
    ↓
後期エポック: stride=1 (精密な時系列学習)
```

### Mixed Sampling Mode
- 10%: Dense sampling (stride=1) - 重要な遷移
- 90%: Sparse sampling (stride=32) - 効率的な学習

## パフォーマンス（Multi-Frame Rate対応）

### 30fpsモード:
- **Mel RTF**: ~0.03 (リアルタイムの3%)
- **Emotion RTF**: ~0.01 (リアルタイムの1%)
- **System RTF**: <0.1 (完全リアルタイム可能)
- **メモリ使用量**: ~355MB (推論時)

### 60fpsモード:
- **Mel RTF**: ~0.05 (リアルタイムの5%)
- **Emotion RTF**: ~0.01 (リアルタイムの1%, 効率的な単一抽出)
- **System RTF**: <0.1 (リアルタイム維持)
- **メモリ使用量**: ~450MB (推論時)

### 共通特性:
- **メモリ効率**: 固定サイズウィンドウによる一定メモリ使用
- **Sequential Processing**: 単一emotion抽出による効率化
- **Dynamic hop_length**: フレームレートに応じた自動計算

## 次のステップ

各ステップの詳細については、上記のリンクから個別のドキュメントを参照してください。
特に重要な変更があったのは：
- [時系列データ読み込み](01_data_loading.md) - Sequential processing
- [デュアルストリーム特徴抽出](03_audio_feature_extraction.md) - 並列特徴抽出
- [デュアルストリームフォワードパス](04_model_forward_pass.md) - 新アーキテクチャ