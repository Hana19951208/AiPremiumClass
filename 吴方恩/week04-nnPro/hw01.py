# 导包
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 初始化数据集
X,y = fetch_olivetti_faces(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32).clamp(0, 1)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2)

# 定义超参数
LR = 1e-3
epochs = 100
BATCH_SIZE = 64

trainDataset = TensorDataset(X_train, y_train)
train_dl = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)  # shuffle=True表示打乱数据

testDataset = TensorDataset(X_test, y_test)
test_dl = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False) 


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

model = NeuralNetwork()

# 定义损失函数
loss_function = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=LR)

# 模型训练
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    # 训练阶段
    for data, target in train_dl:
        optimizer.zero_grad()
        output = model(data.view(-1, 64*64))
        loss = loss_function(output, target)
        # 反向传播
        loss.backward()     # 计算梯度（参数.grad）
        optimizer.step()    # 更新参数
        epoch_loss += loss.item() * data.size(0)
    epoch_loss /= len(train_dl.dataset)
    print(f'Epoch:{epoch} Loss: {epoch_loss:.4f}')

# 模型测试
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data, target in test_dl:
        output = model(data.view(-1,4096))
        _, predicted = torch.max(output, 1)  # 返回每行最大值和索引
        total += target.size(0)  # size(0) 等效 shape[0]
        correct += (predicted == target).sum().item()
print(f'Accuracy: {correct/total*100}%')