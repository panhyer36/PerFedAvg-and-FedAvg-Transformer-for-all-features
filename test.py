"""聯邦學習測試腳本 - 評估訓練好的模型性能

這個腳本提供了完整的模型測試和評估功能，支持：
1. 標準 FedAvg 測試：直接在測試集上評估全局模型
2. Per-FedAvg 個性化測試：使用 validation set 進行個性化適應，在 test set 上評估
3. 多種性能指標計算：MSE、MAE、RMSE、R²、sMAPE
4. 豐富的可視化功能：預測對比圖、時序圖、注意力權重圖等

主要特點：
- 自動識別算法類型（FedAvg vs Per-FedAvg）
- 正確的數據分割：validation set 用於適應，test set 用於評估
- 支持多客戶端批量測試
- 生成詳細的性能報告和可視化結果
- 保存測試結果到 CSV 文件便於後續分析

Per-FedAvg 評估流程：
- 使用 validation set 對全局模型進行個性化適應
- 在完全未見過的 test set 上評估適應後的性能
- 確保評估結果的公正性和可靠性
- 與 FedAvg 使用完全相同的 targets，確保可比較性

數據一致性保證：
- 兩種算法的 targets 來自相同的 trainer.test_model() 方法
- 完整的時間序列可視化，不進行隨機採樣
- 確保評估結果的可靠性和公平比較

"""

import os
import glob
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import load_config
from src.Model import TransformerModel
from src.DataLoader import SequenceCSVDataset
from src.Trainer import FederatedTrainer
import pandas as pd

# 設置matplotlib不顯示圖形，只保存
# 這很重要，因為我們通常在服務器上運行測試，沒有圖形界面
plt.ioff()  # Turn off interactive mode
plt.style.use('default')  # 使用默認樣式確保一致性

def load_trained_model(config, model_path):
    """載入訓練好的模型
    
    這個函數負責：
    1. 根據配置重建 Transformer 模型架構
    2. 從指定路徑載入模型權重
    3. 自動處理設備兼容性（CPU/GPU/MPS）
    
    Args:
        config: 配置對象，包含模型架構參數
        model_path: 模型權重文件路徑
        
    Returns:
        model: 載入權重後的 TransformerModel 實例
        
    Note:
        - 使用 map_location 確保模型可以在不同設備間遷移
        - 如果找不到模型文件，會使用隨機初始化的模型（僅用於調試）
    """
    # 重建模型架構（必須與訓練時完全一致）
    model = TransformerModel(
        feature_dim=config.feature_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        output_dim=config.output_dim,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    ).to(config.device)
    
    # 載入訓練好的權重
    if os.path.exists(model_path):
        # map_location 確保可以在不同設備間載入（如 GPU 訓練的模型在 CPU 上測試）
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using initialized model")
    
    return model

def load_test_data(config):
    """載入測試數據
    
    這個函數負責：
    1. 掃描數據目錄下的所有 CSV 文件
    2. 為每個客戶端創建數據集對象
    3. 使用已保存的標準化器（避免數據洩漏）
    
    重要設計決策：
    - fit_scalers=False：測試時必須使用訓練時保存的標準化器
    - time_based 分割：確保時序數據的正確性
    - 8:1:1 分割比例：與訓練時保持一致
    
    Args:
        config: 配置對象，包含數據路徑和處理參數
        
    Returns:
        test_datasets: 數據集對象列表
        test_names: 對應的客戶端名稱列表
        
    Note:
        如果某個客戶端數據載入失敗，會跳過該客戶端並打印錯誤信息
    """
    test_datasets = []
    test_names = []
    
    # 掃描數據目錄下的所有 CSV 文件
    csv_pattern = os.path.join(config.data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # 按文件名排序，確保結果的一致性
    for csv_file in sorted(csv_files):
        csv_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            # 創建數據集對象
            dataset = SequenceCSVDataset(
                csv_path=config.data_path,
                csv_name=csv_name,
                input_len=config.input_length,    # 輸入序列長度（如 96）
                output_len=config.output_length,   # 輸出序列長度（如 1）
                features=config.features,          # 輸入特徵列表
                target=config.target,              # 目標變量名稱
                save_path=config.data_path,
                train_ratio=0.8,                   # 必須與訓練時一致
                val_ratio=0.1,                     # 必須與訓練時一致
                split_type='time_based',           # 時序數據必須用時間順序分割
                fit_scalers=False                  # 關鍵：使用已保存的標準化器
            )
            
            if len(dataset) > 0:
                test_datasets.append(dataset)
                test_names.append(csv_name)
                print(f"Loaded test dataset {csv_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading {csv_name}: {e}")
    
    return test_datasets, test_names


def personalize_model(model, support_inputs, support_targets, config):
    """使用 validation set 對模型進行個性化適應
    
    這是 Per-FedAvg 測試時的關鍵步驟：
    1. 複製全局模型（避免修改原始模型）
    2. 在 validation set 上進行幾步梯度下降適應
    3. 返回個性化後的模型用於 test set 評估
    
    正確的個性化流程：
    - 訓練時：使用元學習優化「快速適應能力」
    - 測試時：使用 validation set 進行標準梯度下降適應
    - 評估時：在完全未見過的 test set 上評估性能
    
    Args:
        model: 全局模型（將被複製，不會被修改）
        support_inputs: validation set 輸入數據（用於適應）
        support_targets: validation set 目標數據（用於適應）
        config: 配置對象，包含適應學習率和步數
        
    Returns:
        personalized_model: 適應後的模型
        
    Note:
        - 使用 deepcopy 確保不影響原始模型
        - 適應學習率通常設置得較高（如 0.01）以快速適應
        - 適應步數通常較少（如 5-10 步）避免過擬合
    """
    # 創建模型的深拷貝，確保不影響原始全局模型
    from copy import deepcopy
    personalized_model = deepcopy(model)
    personalized_model.train()  # 設置為訓練模式（啟用 dropout 等）
    
    # 設置優化器
    # 注意：這裡使用的學習率通常比訓練時更高，因為我們需要快速適應
    optimizer = torch.optim.Adam(
        personalized_model.parameters(), 
        lr=config.adaptation_lr,      # 適應學習率（如 0.01）
        weight_decay=1e-4             # 保持與訓練時一致的正則化
    )
    criterion = torch.nn.MSELoss()
    
    # 個性化適應步驟
    # 這相當於在新客戶端上進行幾步本地訓練
    for _ in range(config.personalization_steps):
        optimizer.zero_grad()
        outputs = personalized_model(support_inputs)
        loss = criterion(outputs, support_targets)
        loss.backward()
        optimizer.step()
    
    return personalized_model

def evaluate_model_on_dataset(model, dataset, config):
    """在單個數據集上評估模型（標準 FedAvg 評估）
    
    這是標準的聯邦學習評估方法：
    1. 直接使用全局模型在測試集上進行預測
    2. 計算各種性能指標
    3. 不進行任何個性化適應
    
    適用場景：
    - FedAvg 算法的標準評估
    - 評估全局模型的泛化能力
    - 不考慮客戶端個性化需求的情況
    
    Args:
        model: 訓練好的全局模型
        dataset: 客戶端的完整數據集
        config: 配置對象
        
    Returns:
        dict: 包含各種評估指標和預測結果
        
    評估指標說明：
    - MSE：均方誤差，越小越好
    - MAE：平均絕對誤差，更魯棒的誤差度量
    - RMSE：均方根誤差，與目標變量同單位
    - R²：決定係數，衡量模型解釋數據變異的能力（0-1，越大越好）
    """
    # 使用 FederatedTrainer 的工具函數處理數據
    trainer = FederatedTrainer(model, config, config.device)
    
    # 分割數據集（只使用測試集部分）
    _, _, test_dataset = trainer.split_dataset(dataset)
    
    # 創建數據加載器
    _, _, test_loader = trainer.create_data_loaders(None, None, test_dataset)
    
    # 使用 trainer 的測試方法進行評估
    test_loss, predictions, targets = trainer.test_model(test_loader)
    
    # 計算評估指標
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # R² 分數需要特殊處理，避免目標值無變化時的除零錯誤
    if np.var(targets) > 0:
        r2 = r2_score(targets, predictions)
    else:
        r2 = 0.0  # 如果目標值無變化，R² 定義為 0
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }

def evaluate_model_personalized(model, dataset, config):
    """Per-FedAvg 個性化評估
    
    這是 Per-FedAvg 的核心評估方法，模擬實際應用場景：
    1. 使用 validation set 對全局模型進行個性化適應
    2. 在 test set 上評估個性化後的性能
    
    為什麼需要個性化評估？
    - FedAvg 產生的是「平均」模型，可能不適合特定客戶端
    - Per-FedAvg 訓練的模型具有「快速適應」能力
    - 個性化評估能真實反映模型在新客戶端上的表現
    
    正確的評估流程：
    1. 使用 validation set 作為 support set 進行個性化適應
    2. 在 test set 上評估適應後的模型性能
    3. 確保 test set 從未被用於訓練或適應
    
    Args:
        model: Per-FedAvg 訓練的全局模型
        dataset: 客戶端數據集
        config: 配置對象（包含 adaptation_lr、personalization_steps 等）
        
    Returns:
        dict: 評估結果，額外包含 support/query 大小信息
    """
    # 準備數據 - 分別獲取 validation 和 test 數據
    trainer = FederatedTrainer(model, config, config.device)
    _, val_dataset, test_dataset = trainer.split_dataset(dataset)
    _, val_loader, test_loader = trainer.create_data_loaders(None, val_dataset, test_dataset)
    
    # 使用 validation set 作為 support set 進行個性化適應
    support_inputs, support_targets = [], []
    for inputs, targets in val_loader:
        support_inputs.append(inputs)
        support_targets.append(targets)
    
    if not support_inputs:
        print(f"Warning: No validation data available, falling back to standard evaluation")
        return evaluate_model_on_dataset(model, dataset, config)
    
    # 合併所有 validation 批次
    support_inputs = torch.cat(support_inputs, dim=0)
    support_targets = torch.cat(support_targets, dim=0)
    
    # 確保數據在正確的設備上（GPU/CPU）
    support_inputs = support_inputs.to(config.device)
    support_targets = support_targets.to(config.device)
    
    # 個性化適應：使用 validation set (support) 微調模型
    personalized_model = personalize_model(model, support_inputs, support_targets, config)
    
    # 使用統一的 trainer.test_model 方法獲取 test set 結果
    # 這確保與 FedAvg 評估使用完全相同的數據處理流程
    test_loss, predictions_np, targets_np = trainer.test_model(test_loader)
    
    # 注意：上面的 test_loss 是使用原始全局模型計算的
    # 我們需要用個性化模型重新計算 predictions，但保持相同的 targets
    personalized_model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:  # 我們只使用 inputs，targets 已經從 trainer.test_model 獲得
            inputs = inputs.to(config.device)
            outputs = personalized_model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
    
    # 使用個性化模型的預測結果，但保持與 FedAvg 完全相同的 targets
    predictions_np = np.array(all_predictions)
    
    # 重新計算個性化模型的 test_loss
    predictions_tensor = torch.tensor(predictions_np, device=config.device)
    targets_tensor = torch.tensor(targets_np, device=config.device)
    criterion = torch.nn.MSELoss()
    test_loss = criterion(predictions_tensor, targets_tensor).item()
    
    # 計算評估指標
    mse = mean_squared_error(targets_np, predictions_np)
    mae = mean_absolute_error(targets_np, predictions_np)
    rmse = np.sqrt(mse)
    
    # R² 分數（處理目標值無變化的邊界情況）
    if np.var(targets_np) > 0:
        r2 = r2_score(targets_np, predictions_np)
    else:
        r2 = 0.0
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions_np,
        'targets': targets_np,
        'support_size': len(support_inputs),  # validation set 大小（用於適應）
        'query_size': len(targets_np)         # test set 大小（用於評估）
    }

def plot_predictions_vs_targets(predictions, targets, client_name, save_path):
    """繪製預測值vs真實值散點圖
    
    這個散點圖是評估回歸模型的經典可視化方法：
    - 點越接近對角線，預測越準確
    - 點的分散程度反映預測的不確定性
    - 可以直觀發現系統性偏差（如總是高估或低估）
    
    圖中包含：
    1. 散點：每個點代表一個樣本的（真實值，預測值）
    2. 理想線：y=x 的紅色虛線，代表完美預測
    3. 統計信息：MSE 和 R² 分數
    
    Args:
        predictions: 模型預測值數組
        targets: 真實目標值數組  
        client_name: 客戶端名稱（用於圖表標題）
        save_path: 圖片保存路徑
    """
    plt.figure(figsize=(10, 8))
    
    # 繪製散點圖
    # alpha=0.6 使點半透明，便於觀察重疊區域的密度
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    
    # 繪製理想預測線 (y=x)
    # 這條線表示完美預測的情況
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 設置軸標籤和標題
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values - {client_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)  # 添加網格便於讀數
    
    # 在圖上添加統計信息
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions) if np.var(targets) > 0 else 0.0
    
    # 文本框顯示在左上角
    plt.text(0.05, 0.95, f'MSE: {mse:.6f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes,  # 使用相對坐標
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_predictions_vs_targets.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 關閉圖形釋放內存
    print(f"Saved prediction plot to {plot_path}")

def plot_time_series_comparison(predictions, targets, client_name, save_path):
    """繪製時間序列預測對比圖 - 顯示完整數據
    
    時序對比圖展示模型如何跟蹤時間序列的變化：
    - 可以觀察模型是否捕捉到趨勢和週期性
    - 發現預測的滯後或超前現象
    - 識別模型在哪些時間段表現較差
    
    完整數據顯示的優點：
    - 不遺漏任何預測結果，提供完整視角
    - 確保不同算法的可視化結果完全一致
    - 能觀察到所有時間點的預測表現
    - 便於發現局部的預測模式和異常
    
    Args:
        predictions: 預測值數組
        targets: 真實值數組
        client_name: 客戶端名稱
        save_path: 保存路徑
    """
    # 使用所有數據點，不進行採樣
    n_samples = len(predictions)
    
    # 根據數據量調整圖形尺寸
    # 數據點越多，圖形越寬，便於觀察細節
    fig_width = max(15, min(30, n_samples // 100))  # 動態調整寬度，最小15，最大30
    plt.figure(figsize=(fig_width, 6))
    
    # 繪製真實值和預測值
    plt.plot(range(n_samples), targets, 
             label='True Values', linewidth=1.0, alpha=0.8)
    plt.plot(range(n_samples), predictions, 
             label='Predictions', linewidth=1.0, alpha=0.8)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(f'Complete Time Series Prediction Comparison - {client_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加統計信息到圖表
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))
    
    # 在圖表右上角添加統計信息
    stats_text = f'Total samples: {n_samples}\nMSE: {mse:.6f}\nMAE: {mae:.6f}'
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_time_series_complete.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved complete time series plot to {plot_path} (showing all {n_samples} points)")

def plot_attention_weights(model, dataset, client_name, save_path, config, num_samples=5):
    """可視化注意力權重
    
    Transformer 的核心是自注意力機制，可視化注意力權重能幫助理解：
    1. 模型關注序列中的哪些位置
    2. 是否學到了有意義的時間模式
    3. 注意力是否過於分散或集中
    
    生成兩種可視化：
    1. 熱力圖：直觀顯示注意力權重的分布
    2. 疊加圖：將注意力權重與輸入序列疊加，觀察關注點
    
    Args:
        model: Transformer 模型
        dataset: 數據集
        client_name: 客戶端名稱
        save_path: 保存路徑
        config: 配置對象（需包含 show_attention 標誌）
        num_samples: 可視化的樣本數量（默認 5）
        
    Note:
        - 只有當 config.show_attention=True 時才會生成圖表
        - 注意力權重已經過聚合處理（多頭平均）
    """
    # 檢查是否需要顯示注意力權重
    if not config.show_attention:
        return
    
    # 準備數據
    trainer = FederatedTrainer(model, config, config.device)
    _, _, test_dataset = trainer.split_dataset(dataset)
    _, _, test_loader = trainer.create_data_loaders(None, None, test_dataset)
    
    model.eval()  # 評估模式
    sample_count = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            if sample_count >= num_samples:
                break
                
            data = data.to(config.device).float()
            
            # 獲取注意力權重（已在模型中聚合）
            attention_weights = model.get_attention_weights(data)
            
            # 為批次中的每個樣本生成圖表
            for i in range(min(data.size(0), num_samples - sample_count)):
                plt.figure(figsize=(12, 4))
                
                # 子圖1：注意力權重熱力圖
                attn = attention_weights[i].cpu().numpy().squeeze()
                
                plt.subplot(1, 2, 1)
                # 使用熱力圖顯示注意力權重
                # Blues 色彩映射：深藍色表示高注意力
                sns.heatmap(attn.reshape(1, -1), cmap='Blues', cbar=True)
                plt.title(f'Attention Weights - Sample {sample_count + 1}')
                plt.xlabel('Sequence Position')
                
                # 子圖2：輸入序列與注意力權重疊加
                plt.subplot(1, 2, 2)
                input_seq = data[i].cpu().numpy().squeeze()
                positions = range(len(input_seq))
                
                # 繪製輸入序列
                plt.plot(positions, input_seq, 'b-', 
                        label='Input Sequence', alpha=0.7)
                
                # 繪製縮放後的注意力權重
                # 縮放到輸入序列的範圍，便於比較
                plt.plot(positions, attn * np.max(input_seq), 'r-', 
                        label='Attention Weights (scaled)', alpha=0.8)
                
                plt.xlabel('Sequence Position')
                plt.ylabel('Value')
                plt.title(f'Input Sequence with Attention')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(save_path, 
                                       f'{client_name}_attention_sample_{sample_count + 1}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
    
    print(f"Saved attention plots for {client_name}")

def plot_overall_performance(results, save_path):
    """繪製整體性能圖表
    
    這個函數生成一個 2x2 的子圖網格，展示所有客戶端的性能對比：
    1. MSE：均方誤差 - 查看哪些客戶端預測誤差較大
    2. MAE：平均絕對誤差 - 更穩健的誤差度量
    3. RMSE：均方根誤差 - 與目標變量同單位的誤差
    4. R²：決定係數 - 模型解釋能力
    
    通過這個圖可以：
    - 識別表現異常的客戶端（可能需要更多個性化）
    - 評估模型在不同客戶端間的公平性
    - 發現數據分布的異質性
    
    Args:
        results: 字典，鍵為客戶端名稱，值為評估結果
        save_path: 圖表保存路徑
    """
    client_names = list(results.keys())
    metrics = ['mse', 'mae', 'rmse', 'r2']
    
    # 創建 2x2 子圖
    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()  # 展平為一維數組便於迭代
    
    for i, metric in enumerate(metrics):
        # 提取所有客戶端的指定指標值
        values = [results[client][metric] for client in client_names]
        
        # 繪製柱狀圖
        axes[i].bar(range(len(client_names)), values)
        axes[i].set_title(f'{metric.upper()} per Client')
        axes[i].set_xlabel('Client')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # 處理 x 軸標籤
        # 當客戶端數量較少時，顯示所有標籤
        if len(client_names) <= 20:
            axes[i].set_xticks(range(len(client_names)))
            axes[i].set_xticklabels(client_names, rotation=45, ha='right')
        else:
            # 客戶端太多時，只顯示部分標籤避免重疊
            step = len(client_names) // 10  # 大約顯示 10 個標籤
            axes[i].set_xticks(range(0, len(client_names), step))
            axes[i].set_xticklabels([client_names[j] for j in range(0, len(client_names), step)], 
                                  rotation=45, ha='right')
        
        axes[i].grid(True, alpha=0.3)  # 添加網格
    
    plt.tight_layout()  # 自動調整子圖間距
    plot_path = os.path.join(save_path, 'overall_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall performance plot to {plot_path}")

def sMAPE(predictions, targets):
    """計算對稱平均絕對百分比誤差 (Symmetric Mean Absolute Percentage Error)
    
    sMAPE 是 MAPE 的改進版本，解決了以下問題：
    1. 當真實值接近 0 時，MAPE 會趨於無窮大
    2. 對高估和低估的懲罰不對稱
    
    公式：sMAPE = 200% × |預測值 - 真實值| / (|預測值| + |真實值|)
    
    優點：
    - 範圍在 0-200% 之間
    - 對稱性：高估和低估得到相同懲罰
    - 當預測值和真實值都為 0 時定義為 0
    
    Args:
        predictions: 預測值
        targets: 真實值
        
    Returns:
        float: sMAPE 百分比值
    """
    # 特殊情況：兩者都為 0
    if targets == 0 and predictions == 0:
        return 0
    
    # sMAPE 公式
    return 2.0 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets)) * 100

def plot_smape_comparison(predictions, targets, client_name, save_path):
    """儲存實際值與預測值的sMAPE折線圖和直方圖
    
    生成兩種 sMAPE 可視化：
    1. 折線圖：顯示每個樣本的 sMAPE，可以看出誤差的時間模式
    2. 直方圖：顯示 sMAPE 的分布，了解整體誤差情況
    
    通過這些圖可以了解：
    - 是否存在某些時間段預測特別差
    - 誤差分布是否集中（大部分預測都準確）
    - 是否有離群的高誤差預測
    
    Args:
        predictions: 預測值數組
        targets: 真實值數組
        client_name: 客戶端名稱
        save_path: 保存路徑
    """
    # 計算每個樣本的 sMAPE 值
    smape_values = []
    for pred, target in zip(predictions, targets):
        smape_val = sMAPE(pred, target)
        smape_values.append(smape_val)
    
    smape_values = np.array(smape_values)
    
    # 創建包含兩個子圖的圖形
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子圖1：sMAPE 折線圖
    # 顯示誤差隨樣本的變化趨勢
    ax1.plot(range(len(smape_values)), smape_values, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('sMAPE (%)')
    ax1.set_title(f'sMAPE Over Samples - {client_name}')
    ax1.grid(True, alpha=0.3)
    
    # 添加平均值和中位數參考線
    mean_smape = np.mean(smape_values)
    median_smape = np.median(smape_values)
    ax1.axhline(y=mean_smape, color='r', linestyle='--', alpha=0.8, 
                label=f'Mean: {mean_smape:.2f}%')
    ax1.axhline(y=median_smape, color='g', linestyle='--', alpha=0.8, 
                label=f'Median: {median_smape:.2f}%')
    ax1.legend()
    
    # 子圖2：sMAPE 直方圖
    # 顯示誤差的分布情況
    ax2.hist(smape_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('sMAPE (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'sMAPE Distribution - {client_name}')
    ax2.grid(True, alpha=0.3)
    
    # 在直方圖上添加詳細統計信息
    stats_text = (
        f'Mean: {mean_smape:.2f}%\n'
        f'Median: {median_smape:.2f}%\n'
        f'Std: {np.std(smape_values):.2f}%\n'
        f'Min: {np.min(smape_values):.2f}%\n'
        f'Max: {np.max(smape_values):.2f}%'
    )
    
    # 將統計信息放在右上角
    ax2.text(0.75, 0.95, stats_text, 
             transform=ax2.transAxes,  # 使用相對坐標
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_smape_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sMAPE comparison plot to {plot_path}")

def save_results_to_csv(results, save_path):
    """將結果保存為CSV文件並生成統計摘要
    
    保存的 CSV 文件包含：
    - 每個客戶端的所有評估指標
    - 方便後續分析和製圖
    - 可用於比較不同算法或配置的結果
    
    同時在控制台輸出統計摘要：
    - 各指標的平均值 ± 標準差
    - 幫助快速了解整體性能
    
    Args:
        results: 評估結果字典
        save_path: CSV 文件保存路徑
    """
    data = []
    for client_name, metrics in results.items():
        # 創建一行數據
        row = {'client': client_name}
        
        # 添加所有數值指標（排除預測值和目標值數組）
        row.update({k: v for k, v in metrics.items() 
                   if k not in ['predictions', 'targets']})
        data.append(row)
    
    # 轉換為 DataFrame 並保存
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_path, 'test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # 計算並顯示統計摘要
    print("\nTest Results Summary:")
    print("=" * 50)
    
    # 對主要指標計算平均值和標準差
    for metric in ['mse', 'mae', 'rmse', 'r2']:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            
            # 格式化輸出：平均值 ± 標準差
            print(f"{metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")

def main():
    """主測試函數 - 聯邦學習模型評估的入口點
    
    完整的測試流程：
    1. 解析命令行參數（支持自定義配置文件）
    2. 載入訓練好的模型
    3. 載入所有客戶端的測試數據
    4. 根據算法類型選擇評估方法：
       - FedAvg：標準評估
       - Per-FedAvg：個性化評估
    5. 生成豐富的可視化結果
    6. 保存評估結果到 CSV
    
    使用方式：
    ```bash
    # 使用默認配置
    python test.py
    
    # 使用自定義配置
    python test.py --config my_config.yaml
    ```
    """
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Federated Learning Testing')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    # 載入配置
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # 創建保存目錄（如果不存在）
    os.makedirs(config.plot_path, exist_ok=True)
    
    # 載入訓練好的模型
    model_path = os.path.join(config.model_save_path, "final_global_model.pth")
    model = load_trained_model(config, model_path)
    
    # 載入測試數據
    print("Loading test datasets...")
    test_datasets, test_names = load_test_data(config)
    
    if not test_datasets:
        print("No test datasets found!")
        return
    
    # 評估模型性能
    print("Evaluating model performance...")
    results = {}
    
    # 對每個客戶端進行評估
    for dataset, name in zip(test_datasets, test_names):
        print(f"Evaluating on {name}...")
        
        # 根據算法選擇評估方法
        # Per-FedAvg 需要特殊的個性化評估流程
        if hasattr(config, 'algorithm') and config.algorithm == 'per_fedavg':
            print(f"  Using Per-FedAvg personalized evaluation...")
            result = evaluate_model_personalized(model, dataset, config)
            
            # 顯示 validation/test 集合大小信息
            if 'support_size' in result and 'query_size' in result:
                print(f"  Validation set size (for adaptation): {result['support_size']}, "
                     f"Test set size (for evaluation): {result['query_size']}")
        else:
            # 標準 FedAvg 評估
            print(f"  Using standard FedAvg evaluation...")
            result = evaluate_model_on_dataset(model, dataset, config)
        
        results[name] = result
        
        # 顯示評估結果
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  MAE: {result['mae']:.6f}")
        print(f"  RMSE: {result['rmse']:.6f}")
        print(f"  R²: {result['r2']:.4f}")
        
        # 生成可視化圖表
        if config.save_plots:
            print(f"  Generating plots for {name}...")
            
            # 1. 預測值 vs 真實值散點圖
            plot_predictions_vs_targets(
                result['predictions'], result['targets'], 
                name, config.plot_path
            )
            
            # 2. 時間序列對比圖
            plot_time_series_comparison(
                result['predictions'], result['targets'], 
                name, config.plot_path
            )

            # 3. sMAPE 分析圖
            plot_smape_comparison(
                result['predictions'], result['targets'], 
                name, config.plot_path
            )
            
            # 4. 注意力權重可視化
            # 限制只為前幾個客戶端生成，避免產生過多圖片
            if len(results) <= 5:
                plot_attention_weights(
                    model, dataset, name, config.plot_path, config
                )
    
    # 生成整體性能圖表
    if config.save_plots:
        print("Generating overall performance plots...")
        plot_overall_performance(results, config.plot_path)
    
    # 保存結果到 CSV
    save_results_to_csv(results, config.plot_path)
    
    print(f"\nTesting completed! Results saved to {config.plot_path}")

if __name__ == "__main__":
    main()