# FedAvg Transformer 消費者電力需求預測系統

## 項目概述

基於 **聯邦學習（Federated Learning）** 的消費者電力需求預測研究項目，使用 **Transformer 模型** 實現的時序預測。本系統採用 **FedAvg（聯邦平均）算法**。

本系統具備多項專業特性，包括保障用戶數據隱私，所有客戶端數據均保留在本地，僅進行模型參數的共享；採用基於 Transformer 注意力機制的時序建模方法以提升預測效率；支持早停機制與學習率自動調度，優化訓練過程；提供模型性能評估及注意力權重的可視化分析工具；並配備完整的 YAML 配置系統，適用於多種運算設備。


## 快速開始

### 前置需求

- Python 3.8+
- PyTorch 2.0+
- CUDA, MPS（可選，支持 GPU 加速）

### 安裝步驟

1. **克隆項目**
```bash
git clone https://github.com/panhyer36/FedAvgTransformerOPSD.git
cd FedAvgTransformerOPSD
```

2. **安裝依賴**
```bash
pip install -r requirements.txt
```

3. **檢查配置**
```bash
python config.py
```

### 開始訓練

#### 方法一：後台訓練（推薦）
```bash
# 開始訓練
./run.sh

# 監控訓練過程
tail -f logs/train_*.log

# 停止訓練
./stop.sh
```

#### 方法二：直接運行
```bash
# 訓練模型（默認使用 config.yaml）
python train.py

# 訓練模型（指定配置文件）
python train.py --config custom_config.yaml

# 測試和可視化（默認使用 config.yaml）
python test.py

# 測試和可視化（指定配置文件）
python test.py --config custom_config.yaml
```

##  項目架構

### 核心組件

```
src/
├── Server.py          #  聯邦學習協調器（模型聚合、全局訓練）
├── Client.py          #  客戶端實現（本地訓練、驗證評估）
├── Model.py           #  Transformer時序預測模型
├── DataLoader.py      #  時序數據加載器
└── Trainer.py         #  訓練器（早停、學習率調度）
```

### 數據組織

```
data/processed/
├── Consumer_01.csv              # 客戶端數據
├── Consumer_01_*_scaler.pkl     # 標準化器
├── ...                          # Consumer_02 至 Consumer-50
└── Public_Building.csv          # 公共建築數據
```

### 輸出目錄

```
📁 checkpoints/         #  模型檢查點
📁 plots/              #  可視化圖表
📁 logs/               #  訓練日誌
```

##  配置說明

### 核心配置文件：`config.yaml`

#### 聯邦學習配置
```yaml
federated_learning:
  algorithm: "fedavg"        # 聯邦學習算法
  global_rounds: 100         # 全局訓練輪數
  client_fraction: 0.1       # 每輪參與客戶端比例（10%）
  num_clients: 51            # 總客戶端數量
  eval_interval: 10          # 每10輪評估一次
```

#### 模型架構配置
```yaml
model:
  feature_dim: 16            # 輸入特徵維度
  d_model: 512              # Transformer 模型維度
  nhead: 8                  # 多頭注意力數量
  num_layers: 4             # Transformer 層數
  output_dim: 1             # 輸出維度
  dropout: 0.1              # Dropout 比率
```

#### 數據配置
```yaml
data:
  input_length: 96          # 輸入序列長度（96個時間點）
  output_length: 1          # 輸出序列長度（預測1個時間點）
  train_ratio: 0.8          # 訓練集比例（80%）
  val_ratio: 0.1            # 驗證集比例（10%）
  test_ratio: 0.1           # 測試集比例（10%）
```

#### 訓練優化配置
```yaml
training:
  max_epochs: 200           # 最大訓練輪數
  early_stopping:
    patience: 15            # 早停耐心值
    min_delta: 0.001        # 最小改善閾值
  lr_scheduler:
    patience: 5             # 學習率調度耐心值
    factor: 0.5             # 學習率衰減因子
```

## 🔧 使用指南

### 訓練命令詳解

| 命令 | 功能 | 說明 |
|------|------|------|
| `./run.sh` | 後台訓練 | 使用 nohup 後台運行，自動創建日誌 |
| `./stop.sh` | 停止訓練 | 停止訓練進程 |
| `python train.py` | 前台訓練 | 直接在終端運行訓練（默認 config.yaml） |
| `python train.py --config my_config.yaml` | 自定義配置訓練 | 使用指定配置文件進行訓練 |
| `python test.py` | 模型測試 | 加載模型進行測試和可視化（默認 config.yaml） |
| `python test.py --config my_config.yaml` | 自定義配置測試 | 使用指定配置文件進行測試 |

### 監控和日誌

```bash
# 實時查看訓練日誌
tail -f logs/train_*.log

# 檢查訓練狀態
ps aux | grep python

# 查看 GPU 使用情況（如果有 CUDA）
nvidia-smi
```

### 設備配置

系統支持自動設備選擇：
- **CUDA**：NVIDIA GPU（推薦）
- **MPS**：Apple Silicon GPU
- **CPU**：通用 CPU（備用）

## 📊 測試和評估

### 性能指標

運行 `python test.py` 將生成以下評估指標：

- **MSE（均方誤差）**：模型預測精度
- **MAE（平均絕對誤差）**：預測偏差程度
- **RMSE（均方根誤差）**：預測準確性
- **R²（決定係數）**：模型解釋能力

### 可視化輸出

測試完成後，在 `plots/` 目錄中會生成：

1. **預測對比圖**：真實值 vs 預測值散點圖
2. **時序預測圖**：時間序列預測結果對比
3. **注意力權重圖**：Transformer 注意力機制可視化
4. **性能統計圖**：整體模型性能分析

## 技術細節

### 聯邦學習流程

1. **初始化**：Server 創建全局模型並分發給客戶端
2. **本地訓練**：各客戶端在本地數據上訓練模型
3. **參數上傳**：客戶端上傳訓練後的模型參數
4. **模型聚合**：Server 使用 FedAvg 算法聚合參數
5. **模型分發**：更新後的全局模型分發給所有客戶端
6. **迭代訓練**：重複步驟 2-5 直到收斂

### Transformer 模型架構

- **位置編碼**：支持時序信息編碼
- **多頭注意力**：8 個注意力頭並行處理
- **特徵維度**：512 維模型空間
- **層數**：4 層 Transformer 編碼器
- **正則化**：0.1 Dropout 防止過擬合

### 數據處理流程

1. **數據加載**：讀取 CSV 格式的時序電力數據
2. **特徵工程**：16 個電力消費特徵
3. **標準化**：特徵和目標分別標準化
4. **序列切分**：滑動窗口生成訓練樣本
5. **批次組織**：組織成批次供模型訓練

## 🔍 故障排除

### 常見問題

#### 1. 內存不足
```bash
# 減少批次大小
# 修改 config.yaml
local_training:
  batch_size: 16  # 從 32 減少到 16
```

#### 2. CUDA 錯誤
```bash
# 強制使用 CPU
# 修改 config.yaml
device:
  type: "cpu"
```

#### 3. 訓練無法收斂
```bash
# 調整學習率
local_training:
  learning_rate: 0.001  # 從 0.01 降低到 0.001
```

#### 4. 數據加載錯誤
```bash
# 檢查數據完整性
python -c "import pandas as pd; pd.read_csv('data/processed/Consumer_01.csv').info()"
```


## 項目依賴

### 核心依賴

| 套件 | 版本要求 | 功能 |
|------|----------|------|
| torch | ≥2.0.0 | 深度學習框架 |
| numpy | ≥1.21.0 | 數值計算 |
| pandas | ≥1.3.0 | 數據處理 |
| scikit-learn | ≥1.0.0 | 機器學習工具 |
| matplotlib | ≥3.5.0 | 可視化 |
| seaborn | ≥0.11.0 | 統計可視化 |
| pyyaml | ≥6.0 | 配置文件解析 |

### 安裝完整依賴

```bash
pip install -r requirements.txt
```
