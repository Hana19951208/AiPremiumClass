import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

script_dir = os.path.dirname(os.path.abspath(__file__))

# 数据准备
# 数据准备新实现
def load_olivetti():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = torch.FloatTensor(data.data.reshape(-1, 64, 64, 1))  # 转换为64x64图像
    y = torch.LongTensor(data.target)
    
    # 划分训练测试集 (320训练 + 80测试)
    train_X, test_X = X[:320], X[320:]
    train_y, test_y = y[:320], y[320:]
    
    return DataLoader(dataset=list(zip(train_X, train_y)), batch_size=32, shuffle=True), \
           DataLoader(dataset=list(zip(test_X, test_y)), batch_size=32, shuffle=False)

train_loader, test_loader = load_olivetti()


# 创建模型
class FACE_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, label_features, model_name):
        super().__init__()
        if model_name == "rnn":
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_name == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_name == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_name == "birnn":
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True,  bidirectional=True)
        else:
            raise ValueError("model_name must be rnn, lstm, gru, or birnn")
        # 新增双向判断逻辑
        bidirectional = (model_name == "birnn")
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, label_features)  # 修改输入维度
        
        
    def forward(self, input):
        out, _ = self.rnn(input)
        out = self.fc(out[:, -1, :])
        return out

def run(model_name):
    model = FACE_RNN(input_size=64, hidden_size=128, label_features=40, model_name=model_name)

    data_path = os.path.join(script_dir, 'runs', model_name)

    writer = SummaryWriter(data_path)

    # 模型训练
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(300):
        for i, (x, y) in enumerate(train_loader):
            x = x.view(-1, 64, 64)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"ModelName:{model_name}, Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
            writer.add_scalar("transing Loss", loss.item(), epoch * len(train_loader) + i)
                       
        # 模型测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(-1, 64, 64)
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        if epoch % 10 == 0:
            print(f"ModelName:{model_name}, Epoch: {epoch}, Accuracy: {100 * correct / total}%")
            writer.add_scalar("testing Accuracy", 100 * correct / total, epoch)
            

    writer.close()            


if __name__ == "__main__":
    run("rnn")
    run("lstm")
    run("gru")
    run("birnn")