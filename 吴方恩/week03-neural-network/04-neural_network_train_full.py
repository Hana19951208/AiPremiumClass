# 神经网络超参数调试实验（优化版）

## 一、优化后的完整代码

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 实验配置
class Config:
    # 超参数搜索空间
    HIDDEN_SIZES = [64, 128, 256]      # 隐藏层神经元数
    HIDDEN_LAYERS = [1, 2, 3]         # 隐藏层数量
    LEARNING_RATES = [1e-2, 1e-3]     # 学习率
    BATCH_SIZES = [32, 64, 128]       # 批次大小
    EPOCHS = 20                       # 训练轮次
    INPUT_SIZE = 28 * 28                # 输入维度
    NUM_CLASSES = 10                  # 输出类别

# 模型构建器
def build_model(hidden_sizes, num_layers):
    layers = []
    in_features = Config.INPUT_SIZE
    
    # 构建隐藏层
    for i in range(num_layers):
        out_features = hidden_sizes[i] if i < len(hidden_sizes) else hidden_sizes[-1]
        layers += [
            nn.Linear(in_features, out_features),
            nn.ReLU(),                # 改用ReLU激活函数
            nn.BatchNorm1d(out_features)  # 添加批标准化
        ]
        in_features = out_features
    
    # 输出层
    layers.append(nn.Linear(in_features, Config.NUM_CLASSES))
    return nn.Sequential(*layers)

# 训练与评估函数
def train_evaluate(params):
    # 数据加载
    train_set = KMNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_set = KMNIST(root='./data', train=False, transform=ToTensor(), download=True)
    
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False)
    
    # 模型初始化
    model = build_model([params['hidden_size']]*params['num_layers'], params['num_layers'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params['lr']) 
    
    # 训练记录
    history = {'train_loss': [], 'test_acc': []}
    
    # 训练循环
    for epoch in range(Config.EPOCHS):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.view(-1, Config.INPUT_SIZE))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 评估阶段

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.view(-1, Config.INPUT_SIZE))
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 记录指标
        history['train_loss'].append(epoch_loss/len(train_loader))
        history['test_acc'].append(100 * correct / total)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
              f"Loss: {history['train_loss'][-1]:.4f} | "
              f"Acc: {history['test_acc'][-1]:.2f}%")
    
    return history

# 超参数搜索
results = []
for batch_size in Config.BATCH_SIZES:
    for lr in Config.LEARNING_RATES:
        for num_layers in Config.HIDDEN_LAYERS:
            for hidden_size in Config.HIDDEN_SIZES:
                params = {
                    'batch_size': batch_size,
                    'lr': lr,
                    'num_layers': num_layers,
                    'hidden_size': hidden_size
                }
                print(f"\n当前参数组合：{params}")
                history = train_evaluate(params)
                results.append({
                    'params': params,
                    'final_acc': history['test_acc'][-1],
                    'loss_curve': history['train_loss'],
                    'acc_curve': history['test_acc']
                })

# 结果可视化
def plot_results(results):
    plt.figure(figsize=(15, 10))
    
    # 准确率对比
    plt.subplot(2, 1, 1)
    x_labels = [str(r['params']) for r in results]
    acc_values = [r['final_acc'] for r in results]
    plt.bar(range(len(results)), acc_values)
    plt.xticks(range(len(results)), x_labels, rotation=90)
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    
    # 损失曲线对比
    plt.subplot(2, 1, 2)
    for r in results:
        plt.plot(r['loss_curve'], label=str(r['params']))
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

plot_results(results)