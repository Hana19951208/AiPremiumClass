import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os



def plot_series(series, y=None, y_pred=None, y_pred_std=None, x_label="$t$", y_label="$x$"):
    r, c = 3, 5
    fig, axes = plt.subplots(nrows=r, ncols=c, sharey=True, sharex=True, figsize=(20, 10))
    for row in range(r):
        for col in range(c):
            plt.sca(axes[row][col])
            ix = col + row*c
            plt.plot(series[ix, :], ".-")
            if y is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y[ix])), y[ix], "bx", markersize=10)
            if y_pred is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix], "ro")
            if y_pred_std is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] + y_pred_std[ix])
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] - y_pred_std[ix])
            plt.grid(True)
            plt.hlines(0, 0, 100, linewidth=1)
            plt.axis([0, len(series[ix, :])+len(y[ix]), 
             20, np.max(series)+10])  # 修改y轴范围为0到最高温度+10
            if x_label and row == r - 1:
              plt.xlabel(x_label, fontsize=16)
            if y_label and col == 0:
              plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.show()
    fig.savefig(os.path.join(script_dir, 'weather_prediction.png'),  # 使用绝对路径保存
               dpi=300, bbox_inches='tight')  # 设置高分辨率
    plt.close(fig)  # 关闭图形释放内存

class TimeSeriesDataset(Dataset):
  def __init__(self, X, y=None, train=True):
    self.X = X
    self.y = y
    self.train = train

  def __len__(self):
    return len(self.X)

  def __getitem__(self, ix):
    if self.train:
      return torch.from_numpy(self.X[ix]).float().unsqueeze(-1), torch.from_numpy(self.y[ix]).float()
    return torch.from_numpy(self.X[ix]).float().unsqueeze(-1)

from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def fit(model, dataloader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    for epoch in bar:
        model.train()
        train_loss = []
        train_loss2 = []
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_loss2.append((y[:,-1] - y_hat[:,-1]).pow(2).mean().item())
        model.eval()
        eval_loss = []
        eval_loss2 = []
        with torch.no_grad():
            for batch in dataloader['eval']:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                eval_loss.append(loss.item())
                eval_loss2.append((y[:,-1] - y_hat[:,-1]).pow(2).mean().item())
        bar.set_description(f"loss {np.mean(train_loss):.5f} loss_last_step {np.mean(train_loss2):.5f} val_loss {np.mean(eval_loss):.5f} val_loss_last_step {np.mean(eval_loss2):.5f}")
        
def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(device)
        for batch in dataloader:
            X = batch
            X = X.to(device)
            pred = model(X)
            preds = torch.cat([preds, pred])
        return preds
    
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'resources', 'Summary of Weather.csv')
writer_path = os.path.join(script_dir, 'runs', 'weather')
# 参数配置
SEQ_LEN = 30  # 用过去30天预测未来5天
BATCH_SIZE = 32
HIDDEN_SIZE = 64
EPOCHS = 100



# 加载数据
def load_data(filename):
    dates, temps = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row['Date'])
            temps.append(float(row['MaxTemp']))
    return np.array(dates), np.array(temps)

# 创建序列数据集
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data)-seq_len-5):
        x = data[i:i+seq_len]
        y = data[i+seq_len:i+seq_len+5]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 准备数据
dates, temps = load_data(data_path)
scaler = MinMaxScaler()
scaled_temps = scaler.fit_transform(temps.reshape(-1, 1)).flatten()

X, y = create_sequences(scaled_temps, SEQ_LEN)
train_size = int(0.6 * len(X))  # 60% 训练集
valid_size = int(0.8 * len(X))  # 20% 验证集，剩余20%测试集

X_train, X_valid, X_test = X[:train_size], X[train_size:valid_size], X[valid_size:]
Y_train, Y_valid, Y_test = y[:train_size], y[train_size:valid_size], y[valid_size:]


dataset = {
    'train': TimeSeriesDataset(X_train, Y_train),
    'eval': TimeSeriesDataset(X_valid, Y_valid),
    'test': TimeSeriesDataset(X_test, Y_test, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

if __name__ == '__main__':

    model = RNNModel()
    
    fit(model, dataloader)
    y_pred = predict(model, dataloader['test'])
    # 反归一化测试数据
    X_test_denorm = scaler.inverse_transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    Y_test_denorm = scaler.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)
    Y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    # 修改绘图坐标范围
    plot_series(X_test_denorm, Y_test_denorm, Y_pred_denorm)
