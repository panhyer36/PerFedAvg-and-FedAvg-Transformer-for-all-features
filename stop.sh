#!/bin/bash

# 停止聯邦學習訓練腳本

PID_FILE="fedavg_training.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    # 檢查進程是否還在運行
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stopping training process (PID: $PID)..."
        
        # 發送SIGTERM信號
        kill -TERM $PID
        
        # 等待5秒讓進程正常退出
        sleep 5
        
        # 檢查進程是否還在運行
        if ps -p $PID > /dev/null 2>&1; then
            echo "Process still running, sending SIGKILL..."
            kill -KILL $PID
            sleep 2
        fi
        
        # 再次檢查
        if ps -p $PID > /dev/null 2>&1; then
            echo "Failed to stop process $PID"
            exit 1
        else
            echo "Training process stopped successfully."
        fi
    else
        echo "No training process found with PID: $PID"
    fi
    
    # 清理PID文件
    rm -f "$PID_FILE"
    echo "PID file removed."
else
    echo "No PID file found. Training may not be running."
    
    # 嘗試查找Python訓練進程
    TRAIN_PIDS=$(pgrep -f "python.*train.py")
    if [ ! -z "$TRAIN_PIDS" ]; then
        echo "Found training processes: $TRAIN_PIDS"
        echo "Stopping all training processes..."
        echo "$TRAIN_PIDS" | xargs kill -TERM
        sleep 3
        
        # 檢查是否還有殘留進程
        REMAINING_PIDS=$(pgrep -f "python.*train.py")
        if [ ! -z "$REMAINING_PIDS" ]; then
            echo "Force killing remaining processes: $REMAINING_PIDS"
            echo "$REMAINING_PIDS" | xargs kill -KILL
        fi
        echo "All training processes stopped."
    else
        echo "No training processes found."
    fi
fi

echo "Stop script completed."