# PyTorch神经网络训练复习笔记

---

## 一、PyTorch核心概念

### 1. 神经网络模块（`nn.Module`）
- **作用**：定义模型结构的基类，所有网络层和操作需继承此类。
- **关键方法**：
  - `forward()`：定义前向传播逻辑。
  - `parameters()`：返回模型的可训练参数。

### 2. 损失函数
- **常见类型**：
  - `CrossEntropyLoss`：多分类任务。
  - `MSELoss`：回归任务。
- **作用**：量化预测值与真实值的差异，指导反向传播。

### 3. 优化器
- **常见类型**：
  - `SGD`：随机梯度下降。
  - `Adam`：自适应学习率优化器。
- **作用**：通过反向传播更新模型参数，最小化损失。

### 4. 数据加载（`DataLoader`）
- **功能**：
  - 批量加载数据（`batch_size`）。
  - 打乱数据顺序（`shuffle=True`）。
  - 多线程加速（`num_workers`）。

---

## 二、模型构建与训练流程

### 1. 数据准备
```python
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader

# 加载数据集
train_dataset = KMNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = KMNIST(root='./data', train=False, transform=ToTensor(), download=True)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 2. 模型定义
```python
model = nn.Sequential(
    nn.Linear(28*28, 64),  # 输入层 → 隐藏层
    nn.Sigmoid(),          # 激活函数
    nn.Linear(64, 10)      # 隐藏层 → 输出层
)
```

### 3. 训练循环
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    for data, target in train_loader:
        # 前向传播
        y_pred = model(data.reshape(-1, 28*28))
        loss = loss_fn(y_pred, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 三、关键参数对模型的影响

### 1. 神经元数量
- **实验结果**（文件`02`）：
  - 神经元数 `64 → 68.11%`
  - 神经元数 `128 → 68.36%`
  - 神经元数 `256 → 68.12%`
- **结论**：
  - 单纯增加神经元数量对性能提升有限。
  - 可能原因：激活函数（Sigmoid）梯度消失问题，或模型复杂度不足。

### 2. 隐藏层数量
- **文件`02`结构**：
  ```python
  nn.Sequential(
      nn.Linear(784, hidden_size),
      nn.Sigmoid(),
      nn.Linear(hidden_size, 64),  # 新增隐藏层
      nn.Sigmoid(),
      nn.Linear(64, 10)
  )
  ```
- **实验结果**：准确率无显著提升。
- **改进方向**：
  - 使用更深的网络（如3层以上）。
  - 替换激活函数（如ReLU）缓解梯度消失。

### 3. 学习率（LR）
- **实验结果**（文件`03`）：
  - `LR=0.1 → 86.49%`
  - `LR=0.01 → 86.9%`
  - `LR=0.001 → 86.91%`
- **结论**：
  - 学习率过低（如`1e-3`）需配合更大的批次大小。
  - 学习率过高（如`0.1`）可能震荡，需调整优化器或加入学习率衰减。

### 4. 批次大小（Batch Size）
- **实验结果**（文件`03`）：
  - `Batch=32 → 86.49%`
  - `Batch=64 → 86.9%`
  - `Batch=128 → 86.91%`
- **结论**：
  - 较大的批次（如128）可稳定梯度，但需降低学习率。
  - 过大的批次可能导致内存不足或泛化能力下降。

---

## 四、最佳实践总结

| 参数           | 推荐策略                         | 注意事项             |
| -------------- | -------------------------------- | -------------------- |
| **神经元数量** | 适中（如64-256）                 | 结合网络深度调整     |
| **隐藏层数量** | 2-3层（配合ReLU激活）            | 避免过深导致梯度消失 |
| **学习率**     | 初始值`1e-3`，动态调整（如Adam） | 使用学习率调度器     |
| **批次大小**   | 32-128（根据硬件资源选择）       | 大批次需配合低学习率 |

---

## 五、扩展思考
- **激活函数选择**：尝试ReLU或LeakyReLU替代Sigmoid，缓解梯度消失。
- **优化器对比**：测试Adam与SGD的性能差异。
- **正则化**：加入Dropout或L2正则化防止过拟合。