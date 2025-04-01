# 导包
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Windows系统字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 定义超参数
LR = 1e-3
epochs = 100
BATCH_SIZE = 64

# 修改模型定义增加正则化控制
class NeuralNetwork(nn.Module):
    def __init__(self, use_regularization=True):
        super().__init__()
        self.linear1 = nn.Linear(4096, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 40)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1) if use_regularization else nn.Identity()  # 新增正则化开关

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.dropout(x)
        x = self.act(self.linear2(x))
        x = self.dropout(x)
        x = self.act(self.linear3(x))
        return self.linear4(x)

# 新增对比实验函数
def run_experiment(use_normalization, use_regularization):
    # 数据预处理
    X, y = fetch_olivetti_faces(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    if use_normalization:
        X = X.clamp(0, 1)
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 初始化模型
    model = NeuralNetwork(use_regularization=use_regularization)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 训练过程记录
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        for data, target in DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            output = model(data.view(-1, 4096))
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 测试阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE):
                output = model(data.view(-1, 4096))
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 记录指标
        train_losses.append(epoch_loss / len(X_train))
        test_accuracies.append(correct / total)
    
    return train_losses, test_accuracies

# 运行四种实验组合
conditions = [
    (False, False, '无归一化+无正则化'),
    (True, False, '有归一化+无正则化'),
    (False, True, '无归一化+有正则化'), 
    (True, True, '有归一化+有正则化')
]

plt.figure(figsize=(12, 6))

for norm, reg, label in conditions:
    train_loss, test_acc = run_experiment(use_normalization=norm, use_regularization=reg)
    
    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label=label)
    plt.title('训练损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_acc, label=label)
    plt.title('测试准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

plt.legend()
plt.tight_layout()
plt.savefig('training_comparison.png')
plt.show()
