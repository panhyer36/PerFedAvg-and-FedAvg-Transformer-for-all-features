import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy

class Client:
    """
    聯邦學習客戶端類
    
    **支持的算法**：
    1. FedAvg: 標準聯邦平均算法
    2. Per-FedAvg (First-Order): 基於MAML的個性化聯邦學習（一階近似）
    3. Per-FedAvg (HVP): 基於MAML的個性化聯邦學習（二階梯度）
    
    **核心概念**：
    - 元學習(Meta-Learning): 學習如何快速學習的學習算法
    - 內外迴圈: 內迴圈做適應，外迴圈學習好的初始化
    - 支持集/查詢集: 模擬few-shot學習場景的數據分割
    - HVP: 高效計算二階梯度的技術
    
    **使用場景**：
    適用於客戶端數據異質性高，需要個性化模型的聯邦學習場景。
    例如：不同用戶的電力消費模式差異很大，需要個性化預測模型。
    """
    def __init__(self, client_id, train_loader, val_loader, test_loader, model, device, args):
        self.id = client_id
        self.train_loader = train_loader      # 本地訓練數據
        self.val_loader = val_loader          # 本地驗證數據
        self.test_loader = test_loader        # 本地測試數據
        self.model = deepcopy(model).to(device)  # 本地模型副本
        self.device = device                  # 計算設備 (CPU/CUDA/MPS)
        self.args = args                      # 訓練配置參數
        self.criterion = nn.MSELoss()         # 損失函數（回歸任務）


    def train(self):
        """
        標準 FedAvg 本地訓練
        
        **FedAvg 算法原理**：
        1. 每個客戶端在本地數據上訓練模型多個epoch
        2. 客戶端將訓練後的模型參數發送給服務器
        3. 服務器對所有客戶端的參數進行加權平均
        4. 服務器將平均後的全局模型發送回客戶端
        
        **與個性化方法的對比**：
        - FedAvg: 所有客戶端共享同一個全局模型，無個性化
        - Per-FedAvg: 全局模型作為好的初始化，支持快速個性化適應
        
        **訓練流程**：
        就是標準的監督學習：前向傳播 → 計算損失 → 反向傳播 → 更新參數
        """
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.args.beta, weight_decay=1e-4)
        
        # 本地訓練多個 epoch（與 Per-FedAvg 保持一致以便公平比較）
        for _ in range(self.args.tau):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 標準的深度學習訓練步驟
                optimizer.zero_grad()           # 清零梯度
                outputs = self.model(inputs)    # 前向傳播
                loss = self.criterion(outputs, labels)  # 計算損失
                loss.backward()                 # 反向傳播計算梯度
                optimizer.step()                # 更新參數
        
        return self.model.state_dict()  # 返回訓練後的參數，供服務器聚合
    
    def _validate(self):
        """驗證模型性能"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def evaluate_on_validation(self, global_state_dict):
        """用於訓練過程中的評估：載入全局模型並在驗證集上評估"""
        # 載入全局模型狀態
        self.model.load_state_dict(global_state_dict)
        return self._validate()
    
    def evaluate_on_validation_personalized(self, global_state_dict, adaptation_steps=3, support_ratio=0.2):
        """
        Per-FedAvg 個性化驗證評估
        
        **個性化評估的重要性**：
        Per-FedAvg的目標是學習一個能快速適應的全局模型。標準評估無法體現這一點，
        因為它直接使用全局模型，沒有展現個性化適應的效果。
        
        **個性化評估流程**：
        1. 載入全局模型（作為初始化）
        2. 將驗證數據分為support和query兩部分
        3. 在support set上進行少量步驟的適應訓練
        4. 在query set上評估適應後的模型性能
        
        **這模擬了實際部署場景**：
        客戶端收到全局模型後，用少量本地數據快速個性化，然後處理新任務。
        
        **參數說明**：
        - adaptation_steps: 個性化適應的梯度步數（通常1-5步）
        - support_ratio: support set佔驗證數據的比例（通常10-30%）
        """
        # 載入全局模型狀態作為個性化的起點
        self.model.load_state_dict(global_state_dict)
        
        # === 收集所有驗證數據 ===
        all_inputs, all_targets = [], []
        for inputs, targets in self.val_loader:
            all_inputs.append(inputs)
            all_targets.append(targets)
        
        if not all_inputs:
            return float('inf')  # 沒有驗證數據
        
        # 合併所有批次的數據
        all_inputs = torch.cat(all_inputs, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)
        
        total_samples = len(all_inputs)
        support_size = max(1, int(total_samples * support_ratio))
        
        if support_size >= total_samples:
            # 數據太少無法分割，降級到標準評估
            return self._validate()
        
        # === 分割為Support和Query集合 ===
        # 隨機分割模擬真實場景下可用於適應的數據是隨機的
        indices = torch.randperm(total_samples)
        support_indices = indices[:support_size]      # 用於個性化適應
        query_indices = indices[support_size:]        # 用於評估適應效果
        
        support_inputs = all_inputs[support_indices]
        support_targets = all_targets[support_indices]
        query_inputs = all_inputs[query_indices]
        query_targets = all_targets[query_indices]
        
        # === 個性化適應階段 ===
        # 在support set上進行少量步驟的梯度下降
        # 這模擬客戶端收到全局模型後的快速適應過程
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            outputs = self.model(support_inputs)
            loss = self.criterion(outputs, support_targets)
            loss.backward()
            optimizer.step()
            # 注意：這裡每一步都是基於全部support數據，這是few-shot學習的常見做法
        
        # === 評估適應效果 ===
        # 在query set上測試適應後模型的性能
        # 這才是Per-FedAvg真正要優化的指標
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(query_inputs)
            loss = self.criterion(outputs, query_targets)
            total_loss = loss.item()
        
        return total_loss  # 返回個性化適應後的性能
    
    def _get_model_config(self):
        """提取TransformerModel構造所需的配置參數"""
        return {
            'feature_dim': self.model.feature_dim,
            'd_model': self.model.d_model,
            'nhead': getattr(self.model.transformer.layers[0].self_attn, 'num_heads', 8),
            'num_layers': len(self.model.transformer.layers),
            'output_dim': 1,  # 根據實際配置
            'max_seq_length': 100,  # 使用默認值
            'dropout': 0.1  # 使用默認值
        }
    
    def local_train_per_fedavg(self, global_model=None):
        """
        Per-FedAvg 本地訓練 (First-Order MAML近似版本)
        
        **算法原理**：
        Per-FedAvg是基於MAML(Model-Agnostic Meta-Learning)的聯邦學習算法。
        其核心思想是學習一個能夠快速適應各個客戶端的全局模型初始化參數。
        
        **元學習過程**：
        1. 內迴圈(Inner Loop): 在支持集D上做一步梯度更新，獲得適應後的參數
        2. 外迴圈(Outer Loop): 使用適應後的參數在查詢集D'上計算損失，更新原始參數
        
        **與標準FedAvg的區別**：
        - FedAvg: 只在本地數據上訓練，然後平均參數
        - Per-FedAvg: 模擬每個批次的個性化適應過程，學習更好的初始化
        
        **First-Order近似**：
        為了降低計算複雜度，使用First-Order近似避免計算二階梯度
        """
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.args.beta, weight_decay=1e-4)
        
        # 元學習步長 - 控制內迴圈更新的幅度
        # 較小的值(0.01)確保適應過程穩定，不會偏離原始參數太遠
        meta_lr = getattr(self.args, 'meta_step_size', 0.01)
        
        # 本地訓練多個 epoch（與標準FedAvg相同）
        for _ in range(self.args.tau):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # === 元學習數據分割 ===
                # 將批次數據分為兩部分模擬個性化適應場景：
                # D: 支持集(Support Set) - 用於快速適應
                # D': 查詢集(Query Set) - 用於評估適應效果
                if len(inputs) < 2:
                    continue  # 批次太小無法分割，跳過
                
                split_idx = len(inputs) // 2
                inputs_d, labels_d = inputs[:split_idx], labels[:split_idx]        # 支持集
                inputs_d_prime, labels_d_prime = inputs[split_idx:], labels[split_idx:]  # 查詢集
                
                # === 內迴圈：在支持集D上模擬快速適應 ===
                # 創建臨時模型，避免影響原始模型參數
                model_config = self._get_model_config()
                temp_model = type(self.model)(**model_config).to(self.device)
                temp_model.load_state_dict(self.model.state_dict())  # 複製當前參數
                temp_model.train()
                
                # 在支持集上計算損失
                outputs_d = temp_model(inputs_d)
                loss_d = self.criterion(outputs_d, labels_d)
                
                # 計算梯度並執行一步適應更新
                # 這模擬了客戶端收到全局模型後的個性化適應過程
                grads = torch.autograd.grad(loss_d, temp_model.parameters())
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data = param.data - meta_lr * grad  # θ' = θ - α∇L_D(θ)
                
                # === 外迴圈：評估適應效果並更新原始模型 ===
                # 使用適應後的參數在查詢集上計算損失
                outputs_d_prime = temp_model(inputs_d_prime)
                loss_d_prime = self.criterion(outputs_d_prime, labels_d_prime)
                
                # 關鍵：使用查詢集的損失來更新原始模型
                # 這確保原始模型能學到好的初始化，使得一步適應後效果更好
                optimizer.zero_grad()
                loss_d_prime.backward()  # ∇L_D'(θ') 對原始參數θ的梯度
                optimizer.step()
        
        return self.model.state_dict()
    
    def local_train_per_fedavg_hvp(self, global_model=None):
        """
        Per-FedAvg HVP版本 (使用真正的二階梯度)
        
        **與First-Order版本的區別**：
        First-Order版本使用近似來避免計算二階梯度，而HVP版本計算真正的二階梯度。
        
        **HVP (Hessian-Vector Product) 原理**：
        - Hessian矩陣包含所有二階偏導數，描述損失函數的曲率信息
        - 直接計算Hessian矩陣記憶體和計算成本過高（O(n²)）
        - HVP技術只計算Hessian與向量的乘積，大幅降低複雜度至O(n)
        
        **二階梯度的優勢**：
        1. 更準確的梯度信息：考慮損失函數的曲率
        2. 更好的收斂性：避免在鞍點或平坦區域振盪
        3. 更精確的元學習：更好地模擬真實的適應過程
        
        **計算流程**：
        1. 在D上計算一階梯度 ∇L_D(θ)
        2. 計算虛擬更新參數 θ' = θ - α∇L_D(θ)  
        3. 計算L_D'(θ')對原始參數θ的梯度（這是二階梯度！）
        4. 使用此梯度更新原始模型
        """
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.args.beta, weight_decay=1e-4)
        
        # HVP算法參數
        meta_lr = getattr(self.args, 'meta_step_size', 0.01)     # 元學習步長
        hvp_damping = getattr(self.args, 'hvp_damping', 0.01)   # 數值穩定性阻尼
        
        # 本地訓練多個 epoch（與其他版本保持一致）
        for _ in range(self.args.tau):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # === 元學習數據分割 ===
                if len(inputs) < 2:
                    continue  # 批次太小無法分割，跳過
                
                split_idx = len(inputs) // 2
                inputs_d, labels_d = inputs[:split_idx], labels[:split_idx]        # 支持集
                inputs_d_prime, labels_d_prime = inputs[split_idx:], labels[split_idx:]  # 查詢集
                
                # 獲取當前模型參數（用於計算梯度）
                params = list(self.model.parameters())
                
                # === 步驟1：計算支持集上的一階梯度（保持計算圖） ===
                outputs_d = self.model(inputs_d)
                loss_d = self.criterion(outputs_d, labels_d)
                
                # 關鍵：create_graph=True 保持計算圖用於後續二階梯度計算
                # 這使得PyTorch能夠追蹤梯度的梯度
                grads_d = torch.autograd.grad(
                    loss_d, params, 
                    create_graph=True,  # 保持計算圖：讓梯度可以進一步求導
                    retain_graph=True   # 保留圖結構：允許多次backward
                )
                
                # === 步驟2：計算虛擬更新後的參數 ===
                # 這裡不實際更新模型，而是計算如果更新後參數會是什麼
                # θ' = θ - α∇L_D(θ)
                updated_params = [p - meta_lr * g for p, g in zip(params, grads_d)]
                
                # === 步驟3：使用函數式API計算查詢集損失 ===
                # 這是HVP的關鍵：使用虛擬更新的參數計算新的損失
                # 由於updated_params仍在計算圖中，這個損失可以對原始參數求導
                loss_d_prime = self._functional_forward(inputs_d_prime, labels_d_prime, updated_params)
                
                # === 步驟4：計算二階梯度（HVP的核心） ===
                # 計算 ∇_θ L_D'(θ')，這實際上是二階梯度！
                # 因為θ'本身就是θ的函數，所以這包含了二階信息
                hvp_grads = torch.autograd.grad(loss_d_prime, params, retain_graph=False)
                
                # === 步驟5：數值穩定性處理 ===
                # 阻尼技術：添加小的正則項防止數值不穩定
                # 類似於在Hessian矩陣對角線加上小值
                if hvp_damping > 0:
                    hvp_grads = [
                        grad + hvp_damping * param 
                        for grad, param in zip(hvp_grads, params)
                    ]
                
                # === 步驟6：使用二階梯度更新原始模型 ===
                optimizer.zero_grad()
                
                # 手動設置梯度：因為我們計算的是特殊的二階梯度
                # 而不是標準的backward()產生的梯度
                for param, grad in zip(params, hvp_grads):
                    if param.grad is None:
                        param.grad = grad.detach()  # detach切斷計算圖避免內存洩漏
                    else:
                        param.grad.data = grad.detach()
                
                optimizer.step()
        
        return self.model.state_dict()
    
    def _functional_forward(self, inputs, labels, updated_params):
        """
        使用PyTorch函數式API進行前向傳播（HVP關鍵組件）
        
        **為什麼需要函數式API**：
        在HVP計算中，我們需要使用"虛擬更新"的參數來計算損失，但不能實際修改模型參數。
        傳統方法需要：
        1. 保存原始參數
        2. 修改模型參數
        3. 前向傳播
        4. 恢復原始參數
        這樣做會破壞計算圖，無法計算二階梯度。
        
        **torch.func.functional_call的優勢**：
        - 允許使用任意參數執行模型，而不修改模型本身
        - 保持完整的計算圖，支持高階梯度計算
        - 記憶體效率高，避免參數複製和恢復
        - PyTorch 2.0+的現代化功能式編程接口
        
        **工作原理**：
        1. 將updated_params映射到對應的參數名稱
        2. 使用functional_call臨時"替換"模型參數
        3. 執行前向傳播，就像模型真的使用這些參數一樣
        4. 返回的損失仍然在原始計算圖中，可以對原始參數求導
        
        **等價的數學表達**：
        設原始模型為 f(x; θ)，更新後的參數為 θ'
        此函數計算 L(f(x; θ'), y)，其中θ'仍然是θ的函數
        """
        import torch.func as func
        
        # === 構建參數映射字典 ===
        # 將新的參數值映射到對應的參數名稱
        # 這告訴functional_call要用哪些值替換哪些參數
        param_dict = {}
        param_names = [name for name, _ in self.model.named_parameters()]
        
        for name, param in zip(param_names, updated_params):
            param_dict[name] = param  # 'layer.weight' -> 新的weight張量
        
        # === 函數式前向傳播 ===
        # functional_call(model, param_dict, inputs) 等價於：
        # 臨時將model的參數替換為param_dict中的值，然後執行model(inputs)
        # 但保持計算圖完整，不實際修改model的狀態
        outputs = func.functional_call(self.model, param_dict, inputs)
        loss = self.criterion(outputs, labels)
        
        return loss  # 此損失可以對原始參數θ求導！
    
