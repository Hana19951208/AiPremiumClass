# 导包
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 数据准备环节
# 1. 加载数据
train_dataset = KMNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = KMNIST(root='./data', train=False, transform=ToTensor(), download=True)

# 批量大小
BATCH_SIZE = [32,64,128]

# 2.参数设置
# 权重参数
# 输入特征数
input_size = 28*28
# 输出类别数
num_classes = 10
# 神经元数
hidden_size = 64
# 学习率
LR = [1e-1,1e-2,1e-3]
# 迭代次数
epochs = 20
# 定义模型
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.Sigmoid(),
    nn.Linear(hidden_size, num_classes)
)

for idx in range(len(BATCH_SIZE)):
    # print(f'当前批量大小：{BATCH_SIZE[idx]},当前学习率：{LR[idx]}')

    # 数据拆分
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[idx], shuffle=True)
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=LR[idx])

    # 3. 模型训练
    for i in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播(预测值)
            y_hat = model(data.reshape(-1, input_size))
            # 计算损失
            loss = loss_fn(y_hat, target)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f'Epoch {i+1}/{epochs}, Loss: {loss.item()}')

    # 模型评估
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE[idx], shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.reshape(-1, input_size))
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'当前批量大小：{BATCH_SIZE[idx]},当前学习率：{LR[idx]},Accuracy of the network on the 10000 test images: {100 * correct / total}%')


# 当前批量大小：32,当前学习率：0.1,Accuracy of the network on the 10000 test images: 86.49%
# 当前批量大小：64,当前学习率：0.01,Accuracy of the network on the 10000 test images: 86.9%
# 当前批量大小：128,当前学习率：0.001,Accuracy of the network on the 10000 test images: 86.91%