#!/bin/bash

# 聯邦學習訓練和測試腳本
# 使用 nohup 在後台運行

# 檢查Python環境
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Please install Python3."
    exit 1
fi

# 創建必要的目錄
mkdir -p checkpoints
mkdir -p plots
mkdir -p logs

# 設置日志文件
TRAIN_LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
TEST_LOG="logs/test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Federated Learning Training..."
echo "Training log will be saved to: $TRAIN_LOG"
echo "Test log will be saved to: $TEST_LOG"
echo "Use 'tail -f $TRAIN_LOG' to monitor training progress"
echo "Use './stop.sh' to stop training"

# 記錄進程ID
PID_FILE="fedavg_training.pid"

# 後台運行訓練
nohup python3 train.py > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE

echo "Training started with PID: $TRAIN_PID"
echo "PID saved to: $PID_FILE"
echo ""
echo "Training is now running in background."
echo "Script will exit immediately, training continues independently."
echo ""
echo "Note: After training completes, run 'python3 test.py' manually for testing."