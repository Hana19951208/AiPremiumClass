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


# 初始化数据集
X,y = fetch_olivetti_faces(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32).clamp(0, 1)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2)

# 定义超参数
LR = 1e-4
epochs = 500
BATCH_SIZE = 64

trainDataset = TensorDataset(X_train, y_train)
train_dl = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)  # shuffle=True表示打乱数据

testDataset = TensorDataset(X_test, y_test)
test_dl = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False) 
# 定义损失函数
loss_function = nn.CrossEntropyLoss()
# 修改训练部分为对比实验
# 修改优化器配置部分
optimizers = {
    "SGD": (optim.SGD, {"lr": 0.01, "momentum": 0.2}),  # 调整学习率并添加动量
    "RMSprop": (optim.RMSprop, {"lr": LR}),
    "Adam": (optim.Adam, {"lr": LR}),
    "AdamW": (optim.AdamW, {"lr": LR})
}
# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)

        self.linear4 = nn.Linear(128, 40)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.dropout(x)
        x = self.act(self.linear2(x))
        x = self.dropout(x)
        x = self.act(self.linear3(x))
    
        x = self.linear4(x)
        return x
    
results = {}
for opt_name, (opt_class, config) in optimizers.items():
    model = NeuralNetwork()
    # 使用不同优化器的配置参数
    optimizer = opt_class(model.parameters(), **config)
    
    # 训练记录
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        for data, target in train_dl:
            optimizer.zero_grad()
            output = model(data.view(-1, 4096))
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        
        # 测试阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_dl:
                output = model(data.view(-1, 4096))
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 记录结果
        avg_loss = epoch_loss / len(trainDataset)
        train_losses.append(avg_loss)
        test_accuracies.append(correct / total)
        if epoch % 10 == 0:
            print(f"{opt_name} Epoch {epoch}: Loss={avg_loss:.4f}, Acc={correct/total:.2%}")
    
    results[opt_name] = (train_losses, test_accuracies)

# 绘制对比图表
plt.figure(figsize=(14, 6))

# 训练损失对比
plt.subplot(1, 2, 1)
for name, (losses, _) in results.items():
    plt.plot(losses, label=name)
plt.title('训练损失对比')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.legend()

# 测试准确率对比
plt.subplot(1, 2, 2) 
for name, (_, accs) in results.items():
    plt.plot(accs, label=name)
plt.title('测试准确率对比')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.show()
