"""
深度学习综合实验框架（包含数据加载、模型构建、训练测试、可视化对比）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class DeepLearningLab:
    def __init__(self, config):
        # 初始化配置参数
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据预处理
        self._load_data()
        self._build_model()
        print("self model:", self.model)
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))


    def _load_data(self):
        """统一数据加载与预处理"""
        X, y = fetch_olivetti_faces(return_X_y=True)
        X = torch.tensor(X, dtype=torch.float32)
        
        # 归一化处理
        if self.config['normalize']:
            X = (X - X.min()) / (X.max() - X.min())
            
        y = torch.tensor(y, dtype=torch.long)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    def _build_model(self):
        """模块化模型构建"""
        layers = []
        in_features = 4096
        for out_features in [512, 256, 128]:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate'])
            ])
            in_features = out_features
        layers.append(nn.Linear(128, 40))
        
        self.model = nn.Sequential(*layers).to(self.device)
    
    def train(self):
        """综合训练流程"""
        # 初始化优化器和损失函数
        optimizer = self._get_optimizer()
        criterion = nn.CrossEntropyLoss()
        
        # 数据加载器
        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # 训练循环
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for data, target in train_loader:
                data = data.view(-1, 4096).to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 每10个epoch验证一次
            if epoch % 10 == 0:
                val_acc = self.evaluate()
                print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc:.2%}")
    
    def evaluate(self):
        """模型评估"""
        self.model.eval()
        test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 4096).to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
        
        return correct / len(self.X_test)
    
    def _get_optimizer(self):
        """优化器工厂方法"""
        optimizer_type = self.config['optimizer']
        params = list(self.model.parameters())
        optimizers = {
            'sgd': optim.SGD(params, lr=0.01, momentum=0.9),
            'adam': optim.Adam(params, lr=0.001),
            'adamw': optim.AdamW(params, lr=0.001, weight_decay=1e-4),
            'rmsprop': optim.RMSprop(params, lr=0.001)
        }
        return optimizers[optimizer_type.lower()]

class ExperimentRunner:
    """实验对比运行器"""
    @staticmethod
    def run_optimizer_comparison():
        """优化器对比实验"""
        config_template = {
            'epochs': 200,
            'batch_size': 64,
            'normalize': True,
            'dropout_rate': 0.2
        }
        
        results = {}
        for optimizer in ['sgd', 'adam', 'adamw', 'rmsprop']:
            config = config_template.copy()
            config['optimizer'] = optimizer
            lab = DeepLearningLab(config)
            lab.train()
            results[optimizer] = lab.evaluate()
        
        # 绘制对比图表
        plt.figure(figsize=(10, 6))
        plt.bar(results.keys(), results.values())
        plt.title('Optimizer Comparison')
        plt.ylabel('Accuracy')
        plt.savefig('optimizer_comparison1.png')

if __name__ == "__main__":
    # 示例运行配置
    base_config = {
        'epochs': 100,
        'batch_size': 64,
        'normalize': True,
        'dropout_rate': 0.2,
        'optimizer': 'adam'
    }
    
    # 运行优化器对比实验
    ExperimentRunner.run_optimizer_comparison()