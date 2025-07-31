import torch
import numpy as np
from collections import OrderedDict
import os

class Server:
    """
    聯邦學習服務器類 - 協調所有客戶端的訓練過程
    
    **服務器的核心職責**：
    1. 模型聚合：將客戶端的本地模型參數聚合為全局模型
    2. 客戶端選擇：每輪隨機選擇一部分客戶端參與訓練
    3. 訓練協調：控制全局訓練輪數和評估頻率
    4. 早停管理：監控驗證性能，避免過擬合
    
    **聯邦學習的挑戰**：
    - 數據異質性：不同客戶端的數據分佈可能差異很大
    - 通信效率：需要最小化客戶端與服務器間的通信次數
    - 系統異質性：客戶端的計算能力可能不同
    - 隱私保護：不能直接訪問客戶端的原始數據
    
    **支持的聚合算法**：
    - FedAvg: 加權平均聚合（權重通常基於數據量）
    - Per-FedAvg: 針對個性化聯邦學習的特殊聚合
    """
    def __init__(self, global_model, clients, device, args):
        self.global_model = global_model    # 全局共享模型
        self.clients = clients              # 所有客戶端的列表
        self.device = device                # 服務器計算設備
        self.args = args                    # 訓練配置參數

        # 早停機制相關變量 - 防止過擬合
        self.best_val_loss = float('inf')   # 記錄最佳驗證損失
        self.patience_counter = 0           # 當前耐心計數器
        self.early_stop = False             # 早停標誌

    def aggregate_models(self, client_models):
        """
        聯邦平均聚合算法 (FedAvg核心)
        
        **FedAvg聚合原理**：
        對於每個參數層，計算所有參與客戶端對應參數的算術平均值。
        數學表達：θ_global = (1/K) * Σ(θ_client_i)，其中K是參與的客戶端數量
        
        **為什麼使用平均而不是其他聚合方式**：
        1. 理論保證：在凸優化情況下，FedAvg等價於集中式訓練
        2. 通信效率：只需要一次參數傳輸，無需多輪通信
        3. 隱私保護：服務器只看到聚合後的參數，看不到原始數據
        4. 簡單有效：實現簡單，在實踐中表現良好
        
        **聚合過程**：
        1. 收集所有客戶端的模型參數 (state_dict)
        2. 對每個參數層分別計算平均值
        3. 構建新的全局模型參數字典
        
        **數據類型處理**：
        使用.float()確保數值精度，避免半精度訓練造成的精度損失
        """
        avg_state_dict = OrderedDict()
        
        # 遍歷全局模型的每一個參數層
        for k in self.global_model.state_dict().keys():
            # 提取所有客戶端對應層的參數，堆疊後計算平均
            # torch.stack: 將列表中的張量沿新維度堆疊 [client1_param, client2_param, ...] -> [K, param_shape]
            # .mean(0): 沿第0維（客戶端維度）計算平均值 -> [param_shape]
            avg_state_dict[k] = torch.stack([client_model[k].float() for client_model in client_models], 0).mean(0)
        
        return avg_state_dict

    def run(self):
        """
        執行聯邦學習主循環 - 整個FL系統的核心協調邏輯
        
        **聯邦學習訓練流程**：
        1. 客戶端選擇：隨機選擇一部分客戶端參與當前輪次
        2. 模型分發：將當前全局模型發送給選中的客戶端
        3. 本地訓練：客戶端在本地數據上訓練模型
        4. 模型收集：收集客戶端訓練後的模型參數
        5. 模型聚合：將所有客戶端模型聚合為新的全局模型
        6. 評估與早停：定期評估模型性能，決定是否提前停止
        
        **關鍵設計決策**：
        - 部分客戶端參與：每輪只選擇一部分客戶端，提高效率並增加隨機性
        - 異步訓練：不等待所有客戶端，提高系統容錯性
        - 早停機制：防止過擬合，節省計算資源
        """
        for k in range(self.args.K):  # 全局訓練輪數迴圈
            self.global_model.train()
            
            # === 客戶端選擇階段 ===
            # 隨機選擇一部分客戶端參與訓練（典型值：10-100%）
            # 好處：1)減少通信開銷 2)增加隨機性防止過擬合 3)模擬實際部署中的客戶端可用性
            num_active_clients = int(self.args.r * self.args.num_users)
            selected_client_ids = np.random.choice(range(self.args.num_users), num_active_clients, replace=False)
            
            local_models = []  # 存儲客戶端訓練後的模型參數
            
            print(f"Round {k+1}/{self.args.K}")
            
            # === 並行本地訓練階段 ===
            for i, client_id in enumerate(selected_client_ids):
                client = self.clients[client_id]
                
                # 步驟1：將最新的全局模型參數下發給客戶端
                client.model.load_state_dict(self.global_model.state_dict())
                
                # 步驟2：根據配置選擇訓練算法
                if hasattr(self.args, 'algorithm') and self.args.algorithm == 'per_fedavg':
                    # Per-FedAvg 個性化聯邦學習
                    if getattr(self.args, 'use_second_order', False):
                        # HVP版本：使用真正的二階梯度，理論上更準確但計算更昂貴
                        local_model_state = client.local_train_per_fedavg_hvp(self.global_model)
                    else:
                        # First-Order版本：使用一階近似，效率更高
                        local_model_state = client.local_train_per_fedavg(self.global_model)
                else:
                    # 標準 FedAvg：傳統聯邦學習，無個性化
                    local_model_state = client.train()
                
                # 步驟3：收集客戶端訓練後的模型參數
                local_models.append(local_model_state)
                print(f"  Client {client_id+1} training completed ({i+1}/{len(selected_client_ids)})")

            # === 模型聚合階段 ===
            # 將所有客戶端的模型參數聚合為新的全局模型
            global_state_dict = self.aggregate_models(local_models)
            self.global_model.load_state_dict(global_state_dict)

            # === 評估與早停階段 ===
            if (k + 1) % self.args.eval_interval == 0:
                # 在驗證集上評估當前全局模型的性能
                avg_val_loss = self.evaluate()
                
                # 早停邏輯：如果驗證損失有顯著改善，更新最佳模型
                if avg_val_loss < self.best_val_loss - self.args.early_stopping_min_delta:
                    self.best_val_loss = avg_val_loss
                    self.patience_counter = 0
                    
                    # 保存當前最佳模型，用於最終部署
                    torch.save(self.global_model.state_dict(), 
                              os.path.join(self.args.model_save_path, "best_global_model.pth"))
                else:
                    # 驗證損失沒有改善，增加耐心計數器
                    self.patience_counter += 1
                    if self.patience_counter >= self.args.early_stopping_patience:
                        print(f"Early stopping triggered at round {k+1}")
                        self.early_stop = True
                        break  # 提前結束訓練
                
                print(f"Early stop counter: {self.patience_counter}/{self.args.early_stopping_patience}")
            
            if self.early_stop:
                break
        
        # === 訓練完成後的最佳模型載入 ===
        best_model_path = os.path.join(self.args.model_save_path, "best_global_model.pth")
        if os.path.exists(best_model_path):
            self.global_model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Loaded best model with validation loss: {self.best_val_loss:.6f}")

    def evaluate(self):
        """評估全局模型在驗證集上的性能（根據算法選擇評估方式）"""
        self.global_model.eval()
        val_losses = []
        global_state_dict = self.global_model.state_dict()

        for client in self.clients:
            # 根據算法選擇評估方式
            if hasattr(self.args, 'algorithm') and self.args.algorithm == 'per_fedavg':
                # Per-FedAvg: 使用個性化評估（如果客戶端支持）
                if hasattr(client, 'evaluate_on_validation_personalized'):
                    val_loss = client.evaluate_on_validation_personalized(
                        global_state_dict, 
                        adaptation_steps=getattr(self.args, 'personalization_steps', 3),
                        support_ratio=getattr(self.args, 'support_ratio', 0.2)
                    )
                else:
                    # 降級到標準評估
                    val_loss = client.evaluate_on_validation(global_state_dict)
            else:
                # FedAvg: 使用標準評估
                val_loss = client.evaluate_on_validation(global_state_dict)
            
            val_losses.append(val_loss)
        
        avg_val_loss = np.mean(val_losses)
        print(f"\n--- Validation Evaluation ---")
        
        # 顯示詳細的算法信息
        if self.args.algorithm == 'per_fedavg':
            gradient_type = "HVP (Second-Order)" if getattr(self.args, 'use_second_order', False) else "First-Order"
            print(f"Algorithm: PER-FEDAVG ({gradient_type})")
            if getattr(self.args, 'use_second_order', False):
                print(f"HVP Damping: {getattr(self.args, 'hvp_damping', 0.01)}")
            print(f"Meta Step Size: {getattr(self.args, 'meta_step_size', 0.01)}")
        else:
            print(f"Algorithm: {self.args.algorithm.upper()}")
        
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        print(f"-----------------------------\n")
        
        return avg_val_loss
    
