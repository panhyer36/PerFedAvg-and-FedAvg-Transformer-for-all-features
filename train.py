"""
聯邦學習訓練主程序 - 分散式深度學習系統入口
=================================================

本程序實現基於Transformer的聯邦學習系統，用於電力負荷時序預測。
主要特點：
- 分散式學習：多個客戶端本地訓練，服務器協調聚合
- 隱私保護：原始數據不離開客戶端，只共享模型參數
- 算法支持：FedAvg、PER-FedAvg等聯邦學習算法
- 時序建模：基於Transformer的時序預測模型

"""

# === 系統核心依賴庫 ===
import os                                   # 文件和目錄操作
import glob                                 # 文件模式匹配和搜索
import torch                                # PyTorch深度學習框架
import argparse                             # 命令行參數解析
from torch.utils.data import DataLoader    # 數據加載和批次處理
import logging                              # 日誌記錄系統
from dotenv import load_dotenv              # 環境變量管理
import requests                             # HTTP請求支持
# === 專案自定義模組 ===
from config import load_config              # 配置文件載入器
from src.Model import TransformerModel      # Transformer時序預測模型
from src.DataLoader import SequenceCSVDataset  # 時序數據集處理器
from src.Client import Client               # 聯邦學習客戶端實現
from src.Server import Server               # 聯邦學習服務器實現
from src.Trainer import FederatedTrainer   # 聯邦訓練器和工具函數

load_dotenv()

def send_message(message):
    if os.getenv('HOST_LINK') is None:
        return
    url = os.getenv('HOST_LINK')
    name = os.getenv('NAME')
    payload = {
        "name": name,
        "message": message
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

def setup_logging(config):
    """
    配置聯邦學習訓練日誌系統 - 訓練過程監控與調試核心
    
    **日誌系統設計原理**：
    - 雙重輸出：同時輸出到文件和控制台，方便實時監控和後續分析
    - 級別控制：支持不同詳細程度的日誌記錄
    - 時間戳記：每條日誌包含精確時間信息，便於問題追蹤
    - 標準格式：統一的日誌格式，便於自動化分析
    
    **聯邦學習日誌記錄內容**：
    - 訓練進度：全局輪次、客戶端選擇、本地訓練狀態
    - 性能指標：損失值、準確率、收斂情況
    - 系統狀態：設備使用、內存消耗、錯誤警告
    - 模型管理：參數更新、模型保存、檢查點創建

    
    **輸出目標配置**：
    - 文件輸出：持久化日誌到'training.log'，便於後續分析
    - 控制台輸出：實時顯示訓練狀態，便於即時監控
    
    Args:
        config: 配置對象，包含log_level等日誌配置參數
        
    Note:
        - log_level配置項決定日誌的詳細程度
        - 文件日誌會累積所有訓練會話的記錄
        - 控制台輸出便於實時監控訓練進度
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level),           # 從配置獲取日誌級別
        format='%(asctime)s - %(levelname)s - %(message)s', # 標準化日誌格式
        handlers=[
            logging.FileHandler('training.log'),            # 文件輸出處理器
            logging.StreamHandler()                          # 控制台輸出處理器
        ]
    )

def load_client_data(config):
    """
    載入所有客戶端的時序數據 - 聯邦學習數據準備
    
    **聯邦學習數據分佈特點**：
    - 數據異質性：每個客戶端的數據分佈可能差異很大
    - 數據隔離：每個客戶端只能看到自己的數據
    - 數據不平衡：不同客戶端的數據量可能差異巨大
    
    **數據載入策略**：
    1. 發現階段：掃描指定目錄下的所有CSV文件
    2. 驗證階段：檢查每個文件的有效性
    3. 創建階段：為每個有效文件創建時序數據集
    4. 標準化：每個客戶端獨立進行數據標準化
    
    **時序數據集配置**：
    - 時間順序分割：確保沒有數據洩漏
    - 獨立標準化：每個客戶端使用自己的統計信息
    - 滑動窗口：96步輸入→1步輸出的時序預測
    
    Args:
        config: 配置對象，包含數據路徑和參數
        
    Returns:
        tuple: (clients_data, data_files)
        - clients_data: SequenceCSVDataset對象列表
        - data_files: 對應的文件名列表
        
    Raises:
        FileNotFoundError: 指定路徑下沒有找到CSV文件
        ValueError: 沒有有效的客戶端數據
    """
    clients_data = []   # 存儲所有客戶端的數據集對象
    data_files = []     # 存儲對應的文件名
    
    # === 步驟1：發現所有客戶端數據文件 ===
    csv_pattern = os.path.join(config.data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {config.data_path}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # === 步驟2：為每個客戶端創建數據集 ===
    for csv_file in sorted(csv_files):  # 排序確保結果可重現
        # 提取客戶端名稱（如Consumer_01, Public_Building等）
        csv_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            # 為每個客戶端創建獨立的時序數據集
            dataset = SequenceCSVDataset(
                csv_path=config.data_path,
                csv_name=csv_name,
                input_len=config.input_length,      # 96個時間步輸入
                output_len=config.output_length,    # 1個時間步輸出
                features=config.features,           # 25個輸入特徵
                target=config.target,               # Power_Demand目標
                save_path=config.data_path,
                train_ratio=0.8,                    # 80%用於訓練
                val_ratio=0.1,                      # 10%用於驗證
                split_type='time_based',            # 時間順序分割
                fit_scalers=True                    # 使用訓練集擬合標準化器
            )
            
            # 驗證數據集有效性
            if len(dataset) == 0:
                print(f"Warning: Dataset {csv_name} is empty, skipping...")
                continue
            
            clients_data.append(dataset)
            data_files.append(csv_name)
            print(f"Loaded {csv_name}: {len(dataset)} samples")
            
        except Exception as e:
            # 容錯處理：單個客戶端數據有問題不影響整體
            print(f"Error loading {csv_name}: {e}")
            continue
    
    # === 步驟3：驗證至少有一個有效客戶端 ===
    if not clients_data:
        raise ValueError("No valid client data found")
    
    print(f"Successfully loaded {len(clients_data)} client datasets")
    return clients_data, data_files

def create_clients(config, clients_data, data_files, device):
    """
    創建聯邦學習客戶端集合 - 分散式訓練核心組件
    
    **聯邦學習客戶端設計原理**：
    - 數據本地化：每個客戶端只處理自己的數據
    - 模型同構：所有客戶端使用相同的模型架構
    - 獨立訓練：每個客戶端在本地獨立訓練
    - 參數共享：只共享模型參數，不共享原始數據
    
    **客戶端創建流程**：
    1. 模型實例化：為每個客戶端創建獨立的Transformer模型
    2. 數據分割：將時序數據分為訓練/驗證/測試集
    3. 數據加載器：創建高效的批次數據加載器
    4. 客戶端封裝：將模型和數據封裝為Client對象
    
    **數據分割策略**：
    - 時間序列分割：避免數據洩漏，確保預測真實性
    - 訓練集(80%)：用於本地模型訓練
    - 驗證集(10%)：用於模型選擇和早停
    - 測試集(10%)：用於最終性能評估
    
    **內存優化考慮**：
    - 按需加載：使用DataLoader實現批次加載
    - 模型複製：每個客戶端擁有獨立的模型實例
    - 數據本地化：避免跨客戶端數據傳輸
    
    Args:
        config: 配置對象，包含模型和訓練參數
        clients_data: 客戶端數據集列表
        data_files: 對應的數據文件名列表
        device: 計算設備 (cuda/mps/cpu)
        
    Returns:
        list: Client對象列表，每個對象代表一個聯邦學習客戶端
        
    Note:
        - 每個客戶端擁有獨立的模型參數副本
        - 數據加載器配置影響訓練效率
        - 客戶端ID用於聯邦協調和日誌追蹤
    """
    clients = []
    
    # === 為每個數據集創建對應的客戶端 ===
    for client_id, (dataset, file_name) in enumerate(zip(clients_data, data_files)):
        # === 步驟1：創建客戶端本地模型 ===
        # 每個客戶端都有一個獨立的Transformer模型實例
        # 所有客戶端使用相同的架構但獨立的參數
        model = TransformerModel(
            feature_dim=config.feature_dim,          # 輸入特徵維度 
            d_model=config.d_model,                  # Transformer隱藏層維度
            nhead=config.nhead,                      # 多頭注意力頭數
            num_layers=config.num_layers,            # Transformer層數
            output_dim=config.output_dim,            # 輸出維度 (Power_Demand預測)
            max_seq_length=config.max_seq_length,    # 最大序列長度 
            dropout=config.dropout                   # Dropout率用於正則化
        ).to(device)  # 將模型移動到指定設備(GPU/CPU)
        
        # === 步驟2：創建聯邦訓練器並分割數據 ===
        # FederatedTrainer負責本地訓練流程管理
        trainer = FederatedTrainer(model, config, device)
        
        # 時序數據分割：確保時間順序，避免數據洩漏
        # 分割比例：訓練集80% + 驗證集10% + 測試集10%
        train_dataset, val_dataset, test_dataset = trainer.split_dataset(dataset)
        
        # === 步驟3：創建高效數據加載器 ===
        # DataLoader提供批次加載、隨機化、並行處理等功能
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # === 步驟4：封裝客戶端對象 ===
        # Client對象整合模型、數據和訓練邏輯
        client = Client(
            client_id=client_id,              # 客戶端唯一標識符
            train_loader=train_loader,        # 訓練數據加載器
            val_loader=val_loader,            # 驗證數據加載器
            test_loader=test_loader,          # 測試數據加載器
            model=model,                      # 本地模型實例
            device=device,                    # 計算設備
            args=config                       # 訓練配置參數
        )
        
        # === 步驟5：添加到客戶端列表並記錄信息 ===
        clients.append(client)
        print(f"Created client {client_id} ({file_name}): "
              f"Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return clients

def create_global_model(config, device):
    """
    創建聯邦學習全局模型 - 聯邦協調中心核心組件
    
    **全局模型設計原理**：
    - 參數聚合中心：收集並融合所有客戶端的本地更新
    - 架構一致性：與所有客戶端模型保持完全相同的架構
    - 狀態管理：維護全局最優參數狀態
    - 分發機制：向客戶端廣播最新的全局參數
    
    **聯邦學習中的全局模型作用**：
    1. 參數初始化：為所有客戶端提供統一的初始參數
    2. 聚合樞紐：接收客戶端本地更新進行參數聚合
    3. 知識整合：將分散的本地知識整合為全局知識
    4. 性能評估：在測試集上評估全局模型性能
    
    **Transformer架構配置**：
    - 多頭注意力：捕捉時序數據中的複雜依賴關係
    - 位置編碼：處理時間序列的位置信息
    - 層歸一化：穩定訓練過程
    - 殘差連接：緩解梯度消失問題
    
    **模型參數說明**：
    - feature_dim: 輸入特徵維度
    - d_model: Transformer內部表示維度，影響模型表達能力
    - nhead: 多頭注意力頭數，增強並行處理能力
    - num_layers: Transformer層數，決定模型深度
    - max_seq_length: 最大序列長度
    - dropout: 正則化參數，防止過擬合
    
    Args:
        config: 配置對象，包含所有模型架構參數
        device: 計算設備 (cuda/mps/cpu)
        
    Returns:
        TransformerModel: 已初始化的全局模型實例
        
    Note:
        - 全局模型與客戶端模型架構必須完全一致
        - 模型權重初始化會影響聯邦學習收斂性
        - 設備放置影響參數聚合效率
    """
    # === 創建與客戶端架構完全一致的全局模型 ===
    global_model = TransformerModel(
        feature_dim=config.feature_dim,          # 輸入特徵維度 
        d_model=config.d_model,                  # Transformer隱藏層維度
        nhead=config.nhead,                      # 多頭注意力頭數
        num_layers=config.num_layers,            # Transformer層數
        output_dim=config.output_dim,            # 輸出維度 
        max_seq_length=config.max_seq_length,    # 最大序列長度 
        dropout=config.dropout                   # Dropout正則化率
    ).to(device)  # 將模型移動到指定計算設備
    
    return global_model

def main():
    """
    聯邦學習訓練主流程 - 整個系統的入口點
    
    **主流程設計原理**：
    1. 配置驅動：所有參數通過YAML配置文件管理
    2. 模塊化設計：數據載入、客戶端創建、服務器協調分離
    3. 容錯處理：完善的異常處理和日誌記錄
    4. 可重現性：固定隨機種子，結果可重現
    
    **聯邦學習訓練流程**：
    1. 初始化：載入配置、設置日誌、檢查環境
    2. 數據準備：載入所有客戶端數據，創建數據加載器
    3. 模型初始化：創建全局模型和客戶端實例
    4. 聯邦協調：啟動服務器，執行多輪聯邦訓練
    5. 結果保存：保存訓練好的模型和日誌
    
    **錯誤處理策略**：
    - KeyboardInterrupt：優雅處理用戶中斷
    - 數據錯誤：提供清晰的錯誤信息
    - 配置錯誤：自動修正或警告
    """
    # === 步驟1：命令行參數解析 ===
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    # === 步驟2：配置載入和環境設置 ===
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # 設置日誌系統
    setup_logging(config)
    logging.info(f"Starting federated learning with device: {config.device}")
    
    # === 步驟3：數據準備階段 ===
    print("Loading client data...")
    clients_data, data_files = load_client_data(config)
    
    # === 配置驗證和自動修正 ===
    # 檢查實際發現的客戶端數量與配置文件中指定的數量是否一致
    # 這是聯邦學習系統中的重要驗證步驟
    actual_num_clients = len(clients_data)
    if actual_num_clients != config.num_users:
        print(f"Warning: Config specified {config.num_users} clients, but found {actual_num_clients}")
        config.num_users = actual_num_clients  # 自動修正配置以適應實際情況
        # 這樣可以避免因配置不匹配導致的運行時錯誤
    
    # === 步驟4：聯邦學習核心組件創建 ===
    # 按順序創建聯邦學習系統的三個核心組件：客戶端、全局模型、服務器
    
    print("Creating clients...")
    # 為每個數據集創建對應的客戶端，每個客戶端包含本地模型和數據
    clients = create_clients(config, clients_data, data_files, config.device)
    
    print("Creating global model...")
    # 創建全局模型，用於參數聚合和分發
    global_model = create_global_model(config, config.device)
    
    print("Creating server...")
    # 創建聯邦學習服務器，負責協調整個訓練過程
    server = Server(
        global_model=global_model,    # 全局模型實例
        clients=clients,              # 所有客戶端列表
        device=config.device,         # 計算設備
        args=config                   # 訓練配置參數
    )
    
    # === 步驟5：訓練配置顯示和確認 ===
    print(f"Starting federated learning...")
    
    # 根據不同聯邦學習算法顯示特定配置參數
    # 這有助於用戶確認訓練設置是否正確
    if config.algorithm == 'per_fedavg':
        # PER-FedAvg算法特有的配置顯示
        gradient_type = "HVP (Second-Order)" if getattr(config, 'use_second_order', False) else "First-Order"
        print(f"Algorithm: PER-FEDAVG ({gradient_type})")
        print(f"Meta Step Size: {getattr(config, 'meta_step_size', 0.01)}")  # 元學習步長
        if getattr(config, 'use_second_order', False):
            print(f"HVP Damping: {getattr(config, 'hvp_damping', 0.01)}")    # Hessian向量積阻尼參數
    else:
        # 其他算法（如FedAvg）的配置顯示
        print(f"Algorithm: {config.algorithm.upper()}")
    
    # 顯示關鍵訓練參數，便於用戶確認和日誌記錄
    print(f"Global rounds: {config.K}")                                    # 全局訓練輪次
    print(f"Clients per round: {int(config.r * config.num_users)}")        # 每輪參與的客戶端數量
    print(f"Local epochs: {config.tau}")                                   # 本地訓練輪次
    print(f"Learning rate: {config.beta}")                                 # 學習率
    print(f"Device: {config.device}")                                      # 計算設備
    print("-" * 50)  # 分隔線，使輸出更清晰
    
    logging.info("Starting federated learning training...")
    send_message("Starting federated learning training...")
    
    # === 步驟6：執行聯邦學習訓練主循環 ===
    try:
        # 啟動聯邦學習主循環
        # server.run()會執行完整的聯邦學習流程：
        # 1. 客戶端選擇 -> 2. 參數分發 -> 3. 本地訓練 -> 4. 參數聚合
        server.run()
        print("Federated learning completed successfully!")
        send_message("Federated learning completed successfully!")
        
        # === 模型保存和後處理 ===
        if config.save_model:
            # 保存訓練完成的全局模型
            model_path = os.path.join(config.model_save_path, "final_global_model.pth")
            torch.save(global_model.state_dict(), model_path)
            print(f"Final model saved to {model_path}")
            
            # 提示用戶如何進行模型評估
            print(f"Use 'python test.py --config {args.config}' for final evaluation")
            logging.info(f"Final model saved to {model_path}")
            
    except KeyboardInterrupt:
        # 優雅處理用戶中斷（Ctrl+C）
        # 這允許用戶在訓練過程中安全地停止程序
        print("Training interrupted by user")
        logging.info("Training interrupted by user")
        send_message("Training interrupted by user")
    except Exception as e:
        # 處理其他異常，提供詳細的錯誤信息
        print(f"Training failed with error: {e}")
        logging.error(f"Training failed with error: {e}")
        send_message(f"Training failed with error: {e}")
        raise  # 重新拋出異常供調試使用

if __name__ == "__main__":
    """
    程序入口點 - 確保腳本作為主程序運行時執行main函數
    
    這個條件判斷確保：
    1. 直接運行此腳本時，會執行main()函數
    2. 當此模組被其他程序導入時，不會自動執行main()函數
    3. 提供了良好的模組化設計，支持代碼重用
    
    使用方式：
    python train.py --config config.yaml
    """
    main()