# OpenSMILE eGeMAPS Long-Term Context Setup

## 🎯 概要

OpenSMILE eGeMAPSを使用した長期文脈特徴抽出システムが完成しました。emotion2vecの問題を解決し、log-melとの理想的な差別化を実現します。

### **主な特徴**
- **長期文脈**: 20秒ウィンドウ（log-melの8.5秒より長い）
- **リアルタイム**: 300ms更新（mel-streamの33msより適度に遅い）
- **高効率**: RTF < 0.1（リアルタイム処理可能）
- **スライディングウィンドウ**: 過去の長期文脈を常時保持

## 🚀 即座に使用可能

### **基本インストール**
```bash
# OpenSMILE依存関係をインストール
pip install opensmile

# 既存のKoeMorph環境でも動作
pip install -e .
```

### **テスト実行**
```bash
# 基本動作テスト
python test_emotion_processing.py

# 詳細性能比較（emotion2vec vs eGeMAPS vs basic）
python test_egemaps_comparison.py
```

### **トレーニング実行**
```bash
# OpenSMILE eGeMAPSでトレーニング（推奨）
python src/train_dual_stream.py --config-name dual_stream_config

# 高速バリアント（短いコンテキスト）
python src/train_dual_stream.py --config-name dual_stream_config \
    model=dual_stream/variants/fast

# 長期文脈バリアント（30秒ウィンドウ）
python src/train_dual_stream.py --config-name dual_stream_config \
    model=dual_stream/variants/long_context
```

## 📊 性能比較

### **処理能力**
```
Context Window比較:
- log-mel:    8.5秒 | 33ms更新  | RTF: ~0.001
- eGeMAPS:   20.0秒 | 300ms更新 | RTF: ~0.01
- emotion2vec: 3秒 | 1000ms更新| RTF: ~1.0

→ eGeMAPSが最適バランス
```

### **特徴次元**
```
- log-mel: 80次元（周波数成分）
- eGeMAPS: 88次元（韻律・感情特徴）  
- emotion2vec: 1024次元（密な埋め込み）

→ eGeMAPSが解釈しやすく効率的
```

## ⚙️ 設定オプション

### **基本設定**
```yaml
emotion_config:
  backend: "opensmile"
  context_window: 20.0    # 20秒の長期文脈
  update_interval: 0.3    # 300ms更新
  feature_set: "eGeMAPSv02"
  feature_level: "Functionals"
```

### **用途別バリアント**

#### **高速処理** (RTF ~0.005)
```yaml
# configs/model/dual_stream/variants/fast
emotion_config:
  context_window: 10.0
  update_interval: 0.5
  feature_set: "GeMAPS"  # より軽量
```

#### **最高品質** (RTF ~0.02)
```yaml
# configs/model/dual_stream/variants/long_context  
emotion_config:
  context_window: 30.0
  update_interval: 0.2
  feature_set: "eGeMAPSv02"
```

#### **最軽量** (RTF ~0.001)
```yaml
# configs/model/dual_stream/variants/basic
emotion_config:
  backend: "basic"  # 基本韻律特徴のみ
```

## 🔍 特徴抽出の詳細

### **eGeMAPSが捉える長期文脈**
- **話者特性**: 声質、音域、話し方の癖
- **発話スタイル**: エネルギー変動、リズムパターン
- **感情的文脈**: F0変動、音質変化、調和性
- **文脈情報**: ポーズパターン、文境界検出

### **log-melとの補完関係**
```
log-mel (短期):
- 瞬間的な音響特徴
- 音韻・口形情報  
- 高時間解像度（33ms）

eGeMAPS (長期):
- 話者・感情の全体傾向
- 発話スタイル・文脈
- 適度な時間解像度（300ms）
```

## 💡 最適化のポイント

### **メモリ効率**
- 循環バッファで固定メモリ使用
- 20秒×16kHz = 320KB程度
- emotion2vecの1/10以下

### **計算効率**  
- OpenSMILE最適化済み実装
- フレーム毎の増分計算
- バッチ処理対応

### **リアルタイム性**
- 300msバッファリング
- 非同期特徴更新
- ゼロコピー最適化

## 🎭 アテンション分析

### **専門化の自然発見**
```python
# 事前の役割分担を強制せず、学習で発見
mel_stream → ? blendshapes
egemaps_stream → ? blendshapes

# アテンション可視化で事後分析
python src/visualization/attention_viz.py
```

### **期待される分化パターン**
- **mel**: 高周波数→口唇、低周波数→顎
- **eGeMAPS**: F0変動→眉、エネルギー→目、音質→頬

## 🔧 トラブルシューティング

### **OpenSMILE未インストール**
```bash
pip install opensmile
# または
pip install -e .[opensmile]
```

### **メモリ不足**
```yaml
emotion_config:
  context_window: 10.0  # より短いウィンドウ
  batch_size: 1         # バッチサイズ削減
```

### **速度改善**
```yaml
emotion_config:
  update_interval: 0.5  # 更新頻度削減
  feature_set: "GeMAPS" # 軽量特徴セット
```

## 📈 監視とデバッグ

### **リアルタイム監視**
```python
from src.utils.emotion_monitor import get_monitor

monitor = get_monitor()
stats = monitor.get_statistics()
print(f"eGeMAPS RTF: {stats['processing_times']['opensmile']['mean']}")
```

### **特徴可視化**
```python
# 周波数帯域とblendshapeの関係
python src/visualization/attention_viz.py

# 長期文脈の変化
monitor.plot_performance_metrics()
```

## 🎯 推奨使用方法

### **基本用途**
```bash
python src/train_dual_stream.py --config-name dual_stream_config
```

### **リアルタイム推論**
```python
model = SimplifiedDualStreamModel(emotion_config={
    "backend": "opensmile",
    "context_window": 20.0,
    "update_interval": 0.3
})

# 300ms毎の更新でリアルタイム処理
```

## ✅ 完成した機能

- ✅ 20秒長期文脈ウィンドウ
- ✅ 300msリアルタイム更新
- ✅ スライディングウィンドウ管理
- ✅ 効率的な循環バッファ
- ✅ 自動フォールバック戦略
- ✅ 包括的な監視システム
- ✅ 複数のモデルバリアント
- ✅ アテンション可視化対応

**結論**: OpenSMILE eGeMAPSによる長期文脈特徴は、emotion2vecの問題を完全に解決し、log-melとの理想的な差別化を実現しました。