# SimplifiedKoeMorphModel 学習過程の詳細解説

このドキュメントシリーズでは、SimplifiedKoeMorphModelの学習過程を詳細に解説します。
各ステップでデータがどのように処理されるかを具体的に説明します。

## 学習プロセスの概要

SimplifiedKoeMorphModelの学習は以下のステップで構成されています：

1. **[データ読み込み](01_data_loading.md)** - ディスクからデータを読み込む
2. **[バッチ作成](02_batch_creation.md)** - データをバッチ形式に整形
3. **[音声特徴抽出](03_audio_feature_extraction.md)** - 音声からメルスペクトログラムを抽出
4. **[モデルフォワードパス](04_model_forward_pass.md)** - モデルの順伝播処理
5. **[損失計算](05_loss_calculation.md)** - 予測と正解の差を計算
6. **[バックプロパゲーション](06_backpropagation.md)** - 勾配の計算
7. **[パラメータ更新](07_parameter_update.md)** - オプティマイザによる重み更新
8. **[メトリクス計算](08_metrics_calculation.md)** - 評価指標の計算
9. **[ログ記録](09_logging.md)** - TensorBoardへの記録
10. **[チェックポイント保存](10_checkpointing.md)** - モデルの保存

## ファイル構成

```
KoeMorph/
├── src/
│   ├── train.py                 # メイン訓練スクリプト
│   ├── model/
│   │   └── simplified_model.py  # SimplifiedKoeMorphModelの定義
│   ├── data/
│   │   ├── dataset.py          # データセットクラス
│   │   └── io.py               # データ読み込み関数
│   └── model/
│       └── losses.py           # 損失関数の定義
└── output/organized/
    ├── train/                  # 訓練データ
    ├── val/                    # 検証データ
    └── test/                   # テストデータ
```

## データ形式

- **入力音声**: 16kHz, モノラル WAVファイル
- **正解データ**: ARKit 52ブレンドシェイプ値 (JSONLファイル)
- **出力**: 52次元のブレンドシェイプ予測値 [0,1]

## 次のステップ

各ステップの詳細については、上記のリンクから個別のドキュメントを参照してください。