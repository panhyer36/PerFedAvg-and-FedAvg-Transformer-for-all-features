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
plt.ioff()
plt.style.use('default')

def load_trained_model(config, model_path):
    """載入訓練好的模型"""
    model = TransformerModel(
        feature_dim=config.feature_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        output_dim=config.output_dim,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    ).to(config.device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using initialized model")
    
    return model

def load_test_data(config):
    """載入測試數據"""
    test_datasets = []
    test_names = []
    
    csv_pattern = os.path.join(config.data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    for csv_file in sorted(csv_files):
        csv_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            dataset = SequenceCSVDataset(
                csv_path=config.data_path,
                csv_name=csv_name,
                input_len=config.input_length,
                output_len=config.output_length,
                features=config.features,
                target=config.target,
                save_path=config.data_path,
                train_ratio=0.8,
                val_ratio=0.1,
                split_type='time_based',
                fit_scalers=False  # 測試時不重新擬合標準化器
            )
            
            if len(dataset) > 0:
                test_datasets.append(dataset)
                test_names.append(csv_name)
                print(f"Loaded test dataset {csv_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading {csv_name}: {e}")
    
    return test_datasets, test_names

def split_test_data_for_personalization(test_loader, support_ratio=0.2):
    """將測試數據分割為 support 和 query sets"""
    all_inputs, all_targets = [], []
    
    for inputs, targets in test_loader:
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    if not all_inputs:
        return None, None, None, None
    
    # 合併所有批次
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 分割數據
    total_samples = len(all_inputs)
    support_size = int(total_samples * support_ratio)
    
    if support_size == 0:
        support_size = min(1, total_samples)  # 至少一個樣本用於 support
    
    # 隨機選擇 support 樣本
    indices = torch.randperm(total_samples)
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]
    
    support_inputs = all_inputs[support_indices]
    support_targets = all_targets[support_indices]
    query_inputs = all_inputs[query_indices]
    query_targets = all_targets[query_indices]
    
    return support_inputs, support_targets, query_inputs, query_targets

def personalize_model(model, support_inputs, support_targets, config):
    """使用 support set 對模型進行個性化適應"""
    # 創建模型副本進行個性化
    from copy import deepcopy
    personalized_model = deepcopy(model)
    personalized_model.train()
    
    # 設置優化器
    optimizer = torch.optim.Adam(
        personalized_model.parameters(), 
        lr=config.adaptation_lr,
        weight_decay=1e-4
    )
    criterion = torch.nn.MSELoss()
    
    # 個性化適應步驟
    for step in range(config.personalization_steps):
        optimizer.zero_grad()
        outputs = personalized_model(support_inputs)
        loss = criterion(outputs, support_targets)
        loss.backward()
        optimizer.step()
    
    return personalized_model

def evaluate_model_on_dataset(model, dataset, config):
    """在單個數據集上評估模型（標準評估）"""
    trainer = FederatedTrainer(model, config, config.device)
    _, _, test_dataset = trainer.split_dataset(dataset)
    _, _, test_loader = trainer.create_data_loaders(None, None, test_dataset)
    
    test_loss, predictions, targets = trainer.test_model(test_loader)
    
    # 計算評估指標
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # 避免除零錯誤
    if np.var(targets) > 0:
        r2 = r2_score(targets, predictions)
    else:
        r2 = 0.0
    
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
    """Per-FedAvg 個性化評估"""
    trainer = FederatedTrainer(model, config, config.device)
    _, _, test_dataset = trainer.split_dataset(dataset)
    _, _, test_loader = trainer.create_data_loaders(None, None, test_dataset)
    
    # 分割測試數據
    support_inputs, support_targets, query_inputs, query_targets = split_test_data_for_personalization(
        test_loader, config.support_ratio
    )
    
    if support_inputs is None or len(query_inputs) == 0:
        print(f"Warning: Insufficient data for personalization, falling back to standard evaluation")
        return evaluate_model_on_dataset(model, dataset, config)
    
    # 確保數據在正確的設備上
    support_inputs = support_inputs.to(config.device)
    support_targets = support_targets.to(config.device)
    query_inputs = query_inputs.to(config.device)
    query_targets = query_targets.to(config.device)
    
    # 個性化模型
    personalized_model = personalize_model(model, support_inputs, support_targets, config)
    personalized_model.eval()
    
    # 在 query set 上評估個性化模型
    with torch.no_grad():
        predictions = personalized_model(query_inputs)
        criterion = torch.nn.MSELoss()
        test_loss = criterion(predictions, query_targets).item()
    
    # 轉換為 numpy 數組計算指標
    predictions_np = predictions.cpu().numpy().flatten()
    targets_np = query_targets.cpu().numpy().flatten()
    
    # 計算評估指標
    mse = mean_squared_error(targets_np, predictions_np)
    mae = mean_absolute_error(targets_np, predictions_np)
    rmse = np.sqrt(mse)
    
    # 避免除零錯誤
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
        'support_size': len(support_inputs),
        'query_size': len(query_inputs)
    }

def plot_predictions_vs_targets(predictions, targets, client_name, save_path):
    """繪製預測值vs真實值散點圖"""
    plt.figure(figsize=(10, 8))
    
    # 散點圖
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    
    # 理想線 (y=x)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values - {client_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加統計信息
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions) if np.var(targets) > 0 else 0.0
    plt.text(0.05, 0.95, f'MSE: {mse:.6f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_predictions_vs_targets.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction plot to {plot_path}")

def plot_time_series_comparison(predictions, targets, client_name, save_path, max_samples=200):
    """繪製時間序列預測對比圖"""
    # 限制樣本數量以便可視化
    n_samples = min(len(predictions), max_samples)
    idx = np.random.choice(len(predictions), n_samples, replace=False)
    idx = np.sort(idx)
    
    plt.figure(figsize=(15, 6))
    
    plt.plot(range(n_samples), targets[idx], label='True Values', linewidth=1.5, alpha=0.8)
    plt.plot(range(n_samples), predictions[idx], label='Predictions', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(f'Time Series Prediction Comparison - {client_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_time_series.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved time series plot to {plot_path}")

def plot_attention_weights(model, dataset, client_name, save_path, config, num_samples=5):
    """可視化注意力權重"""
    if not config.show_attention:
        return
    
    trainer = FederatedTrainer(model, config, config.device)
    _, _, test_dataset = trainer.split_dataset(dataset)
    _, _, test_loader = trainer.create_data_loaders(None, None, test_dataset)
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            data = data.to(config.device).float()
            
            # 獲取注意力權重
            attention_weights = model.get_attention_weights(data)
            
            for i in range(min(data.size(0), num_samples - sample_count)):
                plt.figure(figsize=(12, 4))
                
                # 注意力權重熱力圖
                attn = attention_weights[i].cpu().numpy().squeeze()
                
                plt.subplot(1, 2, 1)
                sns.heatmap(attn.reshape(1, -1), cmap='Blues', cbar=True)
                plt.title(f'Attention Weights - Sample {sample_count + 1}')
                plt.xlabel('Sequence Position')
                
                # 輸入序列和注意力權重的組合圖
                plt.subplot(1, 2, 2)
                input_seq = data[i].cpu().numpy().squeeze()
                positions = range(len(input_seq))
                
                plt.plot(positions, input_seq, 'b-', label='Input Sequence', alpha=0.7)
                plt.plot(positions, attn * np.max(input_seq), 'r-', 
                        label='Attention Weights (scaled)', alpha=0.8)
                plt.xlabel('Sequence Position')
                plt.ylabel('Value')
                plt.title(f'Input Sequence with Attention')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(save_path, f'{client_name}_attention_sample_{sample_count + 1}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
    
    print(f"Saved attention plots for {client_name}")

def plot_overall_performance(results, save_path):
    """繪製整體性能圖表"""
    client_names = list(results.keys())
    metrics = ['mse', 'mae', 'rmse', 'r2']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[client][metric] for client in client_names]
        
        axes[i].bar(range(len(client_names)), values)
        axes[i].set_title(f'{metric.upper()} per Client')
        axes[i].set_xlabel('Client')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        
        # 設置x軸標籤
        if len(client_names) <= 20:
            axes[i].set_xticks(range(len(client_names)))
            axes[i].set_xticklabels(client_names, rotation=45, ha='right')
        else:
            # 如果客戶端太多，只顯示部分標籤
            step = len(client_names) // 10
            axes[i].set_xticks(range(0, len(client_names), step))
            axes[i].set_xticklabels([client_names[j] for j in range(0, len(client_names), step)], 
                                  rotation=45, ha='right')
        
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'overall_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall performance plot to {plot_path}")

def sMAPE(predictions, targets):
    """計算sMAPE"""
    if targets == 0 and predictions == 0:
        return 0
    return 2.0 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets)) * 100

def plot_smape_comparison(predictions, targets, client_name, save_path):
    """儲存實際值與預測值的sMAPE折線圖(1)和直方圖(2)"""
    # 計算每個樣本的sMAPE值
    smape_values = []
    for pred, target in zip(predictions, targets):
        smape_val = sMAPE(pred, target)
        smape_values.append(smape_val)
    
    smape_values = np.array(smape_values)
    
    # 創建包含兩個子圖的圖形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子圖1：sMAPE折線圖
    ax1.plot(range(len(smape_values)), smape_values, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('sMAPE (%)')
    ax1.set_title(f'sMAPE Over Samples - {client_name}')
    ax1.grid(True, alpha=0.3)
    
    # 添加統計線
    mean_smape = np.mean(smape_values)
    median_smape = np.median(smape_values)
    ax1.axhline(y=mean_smape, color='r', linestyle='--', alpha=0.8, label=f'Mean: {mean_smape:.2f}%')
    ax1.axhline(y=median_smape, color='g', linestyle='--', alpha=0.8, label=f'Median: {median_smape:.2f}%')
    ax1.legend()
    
    # 子圖2：sMAPE直方圖
    ax2.hist(smape_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('sMAPE (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'sMAPE Distribution - {client_name}')
    ax2.grid(True, alpha=0.3)
    
    # 添加統計信息到直方圖
    stats_text = f'Mean: {mean_smape:.2f}%\nMedian: {median_smape:.2f}%\nStd: {np.std(smape_values):.2f}%\nMin: {np.min(smape_values):.2f}%\nMax: {np.max(smape_values):.2f}%'
    ax2.text(0.75, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'{client_name}_smape_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sMAPE comparison plot to {plot_path}")

def save_results_to_csv(results, save_path):
    """將結果保存為CSV文件"""
    data = []
    for client_name, metrics in results.items():
        row = {'client': client_name}
        row.update({k: v for k, v in metrics.items() if k not in ['predictions', 'targets']})
        data.append(row)
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_path, 'test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # 計算並顯示統計摘要
    print("\nTest Results Summary:")
    print("=" * 50)
    for metric in ['mse', 'mae', 'rmse', 'r2']:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"{metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")

def main():
    """主測試函數"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Federated Learning Testing')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    # 載入配置
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # 創建保存目錄
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
    
    for dataset, name in zip(test_datasets, test_names):
        print(f"Evaluating on {name}...")
        
        # 根據算法選擇評估方法
        if hasattr(config, 'algorithm') and config.algorithm == 'per_fedavg':
            print(f"  Using Per-FedAvg personalized evaluation...")
            result = evaluate_model_personalized(model, dataset, config)
            if 'support_size' in result and 'query_size' in result:
                print(f"  Support set size: {result['support_size']}, Query set size: {result['query_size']}")
        else:
            print(f"  Using standard FedAvg evaluation...")
            result = evaluate_model_on_dataset(model, dataset, config)
        
        results[name] = result
        
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  MAE: {result['mae']:.6f}")
        print(f"  RMSE: {result['rmse']:.6f}")
        print(f"  R²: {result['r2']:.4f}")
        
        # 生成可視化圖表
        if config.save_plots:
            print(f"  Generating plots for {name}...")
            
            # 預測值vs真實值散點圖
            plot_predictions_vs_targets(
                result['predictions'], result['targets'], 
                name, config.plot_path
            )
            
            # 時間序列對比圖
            plot_time_series_comparison(
                result['predictions'], result['targets'], 
                name, config.plot_path
            )

            # 繪製sMAPE圖表
            plot_smape_comparison(
                result['predictions'], result['targets'], 
                name, config.plot_path
            )
            
            # 注意力權重可視化（僅針對前幾個客戶端，避免生成太多圖片）
            if len(results) <= 5:
                plot_attention_weights(
                    model, dataset, name, config.plot_path, config
                )
    
    # 生成整體性能圖表
    if config.save_plots:
        print("Generating overall performance plots...")
        plot_overall_performance(results, config.plot_path)
    
    # 保存結果到CSV
    save_results_to_csv(results, config.plot_path)
    
    print(f"\nTesting completed! Results saved to {config.plot_path}")

if __name__ == "__main__":
    main()