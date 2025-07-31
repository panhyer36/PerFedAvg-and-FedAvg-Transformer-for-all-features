import yaml
import torch
import os
from types import SimpleNamespace

def load_config(config_path="config.yaml"):
    """
    從YAML配置文件載入並解析所有訓練參數
    
    **配置系統設計原理**：
    1. 集中配置：所有參數集中在一個YAML文件中，便於管理
    2. 參數驗證：載入時進行基本的參數檢查和轉換
    3. 設備自動選擇：自動檢測並選擇最佳的計算設備
    4. 目錄創建：自動創建必要的輸出目錄
    
    **為什麼使用YAML而不是JSON**：
    - 支持註釋：可以在配置文件中添加說明
    - 更易讀：層次結構更清晰
    - 數據類型豐富：支持布爾值、浮點數等
    
    **為什麼轉換為SimpleNamespace**：
    - 屬性訪問：config.lr 比 config['lr'] 更簡潔
    - IDE支持：更好的代碼提示和檢查
    - 一致性：整個項目使用統一的配置訪問方式
    
    Args:
        config_path: YAML配置文件路徑
        
    Returns:
        SimpleNamespace: 包含所有配置參數的對象
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: YAML格式錯誤
        RuntimeError: 其他讀取錯誤
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading configuration file {config_path}: {e}")
    
    # 創建命名空間對象來存儲配置，支持點號訪問
    config = SimpleNamespace()
    
    # === 聯邦學習核心配置 ===
    fl_config = config_dict['federated_learning']
    config.algorithm = fl_config['algorithm']          # 算法選擇：'fedavg' 或 'per_fedavg'
    config.K = fl_config['global_rounds']              # 全局訓練輪數
    config.r = fl_config['client_fraction']            # 每輪參與的客戶端比例
    config.num_users = fl_config['num_clients']        # 總客戶端數量
    config.eval_interval = fl_config['eval_interval']  # 評估間隔（每N輪評估一次）
    
    # === 本地訓練配置 ===
    local_config = config_dict['local_training']
    config.tau = local_config['local_epochs']          # 本地訓練輪數（客戶端）
    config.beta = local_config['learning_rate']        # 學習率
    config.batch_size = local_config['batch_size']     # 批次大小
    
    # === Per-FedAvg 特殊配置 ===
    # 只有當選擇per_fedavg算法時才載入這些參數
    if config.algorithm == 'per_fedavg' and 'per_fedavg' in fl_config:
        per_fedavg_config = fl_config['per_fedavg']
        config.meta_step_size = per_fedavg_config['meta_step_size']                    # 元學習步長
        config.use_second_order = per_fedavg_config.get('use_second_order', False)    # 是否使用HVP
        config.hvp_damping = per_fedavg_config.get('hvp_damping', 0.01)               # HVP數值穩定性
    else:
        # 為其他算法提供默認值，避免AttributeError
        config.meta_step_size = 0.01
        config.use_second_order = False
        config.hvp_damping = 0.01
    
    # 模型配置
    model_config = config_dict['model']
    config.feature_dim = model_config['feature_dim']
    config.d_model = model_config['d_model']
    config.nhead = model_config['nhead']
    config.num_layers = model_config['num_layers']
    config.output_dim = model_config['output_dim']
    config.max_seq_length = model_config['max_seq_length']
    config.dropout = model_config['dropout']
    
    # 数据配置
    data_config = config_dict['data']
    config.data_path = data_config['data_path']
    config.input_length = data_config['input_length']
    config.output_length = data_config['output_length']
    config.features = data_config['features']
    config.target = data_config['target']
    config.train_ratio = data_config['train_ratio']
    config.val_ratio = data_config['val_ratio']
    config.test_ratio = data_config['test_ratio']
    
    # 训练配置
    training_config = config_dict['training']
    config.max_epochs = training_config['max_epochs']
    config.early_stopping_patience = training_config['early_stopping']['patience']
    config.early_stopping_min_delta = training_config['early_stopping']['min_delta']
    config.lr_scheduler_patience = training_config['lr_scheduler']['patience']
    config.lr_scheduler_factor = training_config['lr_scheduler']['factor']
    config.lr_scheduler_min_lr = training_config['lr_scheduler']['min_lr']
    
    # 设备配置
    device_config = config_dict['device']
    config.device = get_device(device_config['type'])
    
    # 日志配置
    logging_config = config_dict['logging']
    config.log_interval = logging_config['log_interval']
    config.save_model = logging_config['save_model']
    config.model_save_path = logging_config['model_save_path']
    config.log_level = logging_config['log_level']
    
    # 测试配置
    testing_config = config_dict.get('testing', {})
    config.personalization_steps = testing_config.get('personalization_steps', 3)
    config.support_ratio = testing_config.get('support_ratio', 0.2)
    config.adaptation_lr = testing_config.get('adaptation_lr', 0.001)
    
    # 可视化配置
    viz_config = config_dict['visualization']
    config.save_plots = viz_config['save_plots']
    config.plot_path = viz_config['plot_path']
    config.show_attention = viz_config['show_attention']
    config.save_only = viz_config['save_only']
    
    # 创建必要的目录
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.plot_path, exist_ok=True)
    
    return config

def get_device(device_type):
    """
    智能設備選擇函數 - 根據配置和硬件可用性選擇最佳計算設備
    
    **設備優先級策略**：
    1. CUDA (NVIDIA GPU): 最快，適合大規模訓練
    2. MPS (Apple Silicon): 在M1/M2 Mac上提供GPU加速
    3. CPU: 通用兼容，但速度較慢
    
    **自動選擇邏輯** (device_type="auto"):
    - 優先選擇CUDA（如果可用）
    - 其次選擇MPS（在Apple Silicon Mac上）
    - 最後降級到CPU
    
    **為什麼這樣設計**：
    - 性能優化：自動選擇最快的可用設備
    - 跨平台兼容：支持不同操作系統和硬件
    - 容錯處理：設備不可用時自動降級
    
    **聯邦學習中的設備考慮**：
    - 服務器：通常使用CUDA GPU
    - 客戶端：可能是CPU、MPS或小型GPU
    - 混合環境：不同客戶端可能使用不同設備
    
    Args:
        device_type: 設備類型字符串 ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: PyTorch設備對象
    """
    if device_type == "auto":
        # 自動選擇：按性能優先級檢查設備可用性
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    elif device_type in ["cuda", "mps", "cpu"]:
        # 指定設備類型：檢查是否可用，不可用則降級到CPU
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        # 無效的設備類型：警告並使用CPU
        print(f"Invalid device type: {device_type}, using CPU")
        return torch.device("cpu")

if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    print(f"Device: {config.device}")
    print(f"Global rounds: {config.K}")
    print(f"Number of clients: {config.num_users}")