# ステップ4: モデルフォワードパス (Model Forward Pass)

## 概要
メル特徴を入力として、クロスアテンション機構を使用してARKitブレンドシェイプ値を予測します。

## フォワードパスの流れ

### 1. モデル呼び出し
**実装場所**: `src/train.py:182`

```python
pred_blendshapes = self.model(audio)  # SimplifiedKoeMorphModel.forward()
```

**入力**: 
- audio: `torch.Size([16, 160000])` (バッチ音声データ)

### 2. 音声エンコーディング
**実装場所**: `src/model/simplified_model.py:130` 

```python
# メル特徴抽出（前ステップで説明済み）
mel_features = self.extract_mel_features(audio)  # (16, 300, 80)

# 音声エンコーダに通す
audio_encoded = self.audio_encoder(mel_features)  # (16, 300, 256)
```

#### 音声エンコーダの構造
**実装場所**: `src/model/simplified_model.py:44-51`

```python
self.audio_encoder = nn.Sequential(
    nn.Linear(80, 256),      # 80次元 → 256次元
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 256),     # 256次元 → 256次元
    nn.ReLU(),
    nn.Dropout(0.1),
)
```

**処理の詳細**:
1. 線形変換1: メル特徴(80次元) → 隠れ表現(256次元)
2. ReLU活性化: 負の値を0に
3. Dropout: 10%のユニットをランダムに無効化（過学習防止）
4. 線形変換2: 特徴をさらに変換
5. ReLU活性化
6. Dropout

**出力**: `torch.Size([16, 300, 256])` (各時間フレームが256次元ベクトル)

### 3. ブレンドシェイプクエリの準備
**実装場所**: `src/model/simplified_model.py:75-77, 133`

```python
# 学習可能なクエリベクトル（初期化時）
self.blendshape_queries = nn.Parameter(
    torch.randn(52, 256) * 0.1  # 52個のブレンドシェイプ用クエリ
)

# バッチサイズ分複製
queries = self.blendshape_queries.unsqueeze(0).repeat(16, 1, 1)
# Shape: (16, 52, 256)
```

**クエリの意味**:
- 各クエリは特定のブレンドシェイプ（例：eyeBlinkLeft）に対応
- 学習により、対応する音声特徴を選択的に注目するように最適化

### 4. クロスアテンション
**実装場所**: `src/model/simplified_model.py:136-141`

```python
attn_output, _ = self.attention(
    query=queries,        # (16, 52, 256) - 何を探すか
    key=audio_encoded,    # (16, 300, 256) - どこから探すか
    value=audio_encoded,  # (16, 300, 256) - 何を取得するか
    need_weights=False
)
```

#### アテンション機構の詳細
**設定**: `num_heads=8, embed_dim=256`

**処理フロー**:
1. **Query投影**: queries → Q (8ヘッド × 32次元)
2. **Key/Value投影**: audio_encoded → K, V
3. **アテンションスコア計算**:
   ```
   scores = Q @ K.T / sqrt(32)  # スケーリング
   attention_weights = softmax(scores)
   ```
4. **値の集約**:
   ```
   output = attention_weights @ V
   ```

**出力**: `torch.Size([16, 52, 256])`
- 各ブレンドシェイプが音声の全時間から関連情報を収集

### 5. ブレンドシェイプデコード
**実装場所**: `src/model/simplified_model.py:144`

```python
blendshapes = self.decoder(attn_output)  # (16, 52, 52)
```

#### デコーダの構造
**実装場所**: `src/model/simplified_model.py:62-72`

```python
self.decoder = nn.Sequential(
    nn.Linear(256, 128),     # 256 → 128次元
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 128),     # 128 → 128次元
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 52),      # 128 → 52次元
    nn.Sigmoid()             # [0, 1]に制限
)
```

**なぜSigmoid？**: ブレンドシェイプ値は0（無表情）から1（最大変形）の範囲

### 6. 次元削減
**実装場所**: `src/model/simplified_model.py:147`

```python
blendshapes = blendshapes.mean(dim=1)  # (16, 52)
```

**理由**: 
- アテンション出力は(16, 52, 52)の3次元
- 各ブレンドシェイプクエリが52次元の出力を生成
- 平均化により最終的な52次元ブレンドシェイプ値を取得

## データフローまとめ

```
入力音声 (16, 160000)
    ↓ メル特徴抽出
メル特徴 (16, 300, 80)
    ↓ 音声エンコーダ
音声表現 (16, 300, 256)
    ↓ クロスアテンション（クエリ: 52×256）
注目済み特徴 (16, 52, 256)
    ↓ デコーダ
生ブレンドシェイプ (16, 52, 52)
    ↓ 平均化
最終出力 (16, 52)
```

## 計算グラフとメモリ

### パラメータ数
- 音声エンコーダ: 80×256 + 256×256 = 86,016
- ブレンドシェイプクエリ: 52×256 = 13,312
- アテンション: 256×256×3×8 = 1,572,864
- デコーダ: 256×128 + 128×128 + 128×52 = 55,808
- **合計**: 約1.73Mパラメータ

### メモリ使用量（フォワードパス）
- 中間活性化: 約50MB（バッチサイズ16）
- 勾配保存: 約100MB（バックプロパゲーション用）

## まとめ

このステップでは：
1. メル特徴を256次元の音声表現にエンコード
2. 52個の学習可能なクエリで音声情報を検索
3. クロスアテンションで各ブレンドシェイプに関連する音声特徴を抽出
4. デコーダで最終的なブレンドシェイプ値（0-1）を生成
5. SimplifiedModelでは単一フレーム出力のため次元削減

次のステップでは、予測値と正解値の差を計算する損失関数について説明します。