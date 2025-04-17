import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import pickle
import os
import jieba
import numpy as np
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度

# 1. 模型定义
class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output

# 2. 数据加载和预处理
def load_datasets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    datasets = {}
    
    # 加载两种分词数据集
    for name in ['jieba', 'spm']:
        with open(os.path.join(base_path, f'data/comments_{name}.pkl'), 'rb') as f:
            datasets[name] = pickle.load(f)
    return datasets

step_losses = []  # 新增：记录每一步的损失

# 3. 模型训练函数
def train_model(dataset, model_name):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建词汇表
    vocab = build_from_doc(dataset)
    comms, lables = convert_data(dataset,vocab)
    
    # 将comms和lables转换成TensorDataset
    dataset = TensorDataset(comms, lables)
    
    # 模型配置
    model = Comments_Classifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_size=256,
        num_classes=2
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 数据加载
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # 训练记录
    losses = []
    
    print(f'\n=== 开始训练 {model_name} 模型 ===')
    
    for epoch in range(5):
        epoch_loss = 0.0
        for step, (inputs, labels) in enumerate(dataloader, 1):  # 从1开始计数
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 记录并打印每一步的损失
            step_loss = loss.item()
            step_losses.append(step_loss)
            epoch_loss += step_loss
            print(f'Epoch [{epoch+1}/10] Step [{step}/{len(dataloader)}] Loss: {step_loss:.4f}')
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}: 平均损失 {avg_loss:.4f}')
    
    return model, vocab, {'epoch': losses, 'step': step_losses}  # 修改返回结构

def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx
# 数据转换
def convert_data(batch_data,vocab):
    comments, votes = [],[]
    # 分别提取评论和标签
    for comment, vote in batch_data:
        comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
        votes.append(vote)
    
    # 将评论和标签转换为tensor
    commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])  # 填充为相同长度
    labels = torch.tensor(votes, dtype=torch.long)
    # 返回评论和标签
    return commt, labels

# 4. 预测和可视化
def compare_models(models, vocabs):
    # 测试样本
    test_samples = [
        "这部电影太精彩了，演员演技炸裂！",
        "完全看不懂在演什么，浪费钱",
        "特效不错，但剧情有些拖沓"
    ]
    
    # 结果对比
    results = []
    
    for text in test_samples:
        # Jieba处理
        jieba_tokens = jieba.lcut(text)
        jieba_ids = [vocabs['jieba'].get(word, 1) for word in jieba_tokens]
        
        # SPM处理（需要先加载SPM模型）
        spm_tokens = jieba.lcut(text)  # 此处应为SPM分词，简化为jieba
        spm_ids = [vocabs['spm'].get(word, 1) for word in spm_tokens]
        
        # 转换为Tensor
        jieba_input = torch.tensor([jieba_ids]).to(models['jieba'].device)
        spm_input = torch.tensor([spm_ids]).to(models['spm'].device)
        
        # 获取预测
        with torch.no_grad():
            jieba_pred = torch.softmax(models['jieba'](jieba_input), dim=1)
            spm_pred = torch.softmax(models['spm'](spm_input), dim=1)
        
        results.append({
            'text': text,
            'jieba': jieba_pred.cpu().numpy(),
            'spm': spm_pred.cpu().numpy()
        })
    
    # 打印结果
    print("\n预测结果对比:")
    print("="*60)
    for res in results:
        print(f"文本: {res['text']}")
        print(f"Jieba预测: {np.argmax(res['jieba'])} (置信度: {np.max(res['jieba']):.2f})")
        print(f"SPM预测: {np.argmax(res['spm'])} (置信度: {np.max(res['spm']):.2f})")
        print("-"*60)

if __name__ == '__main__':
    # 加载数据
    datasets = load_datasets()
    
    # 训练两个模型
    models = {}
    vocabs = {}
    losses = {}
    
    # 训练Jieba模型
    model_jieba, vocab_jieba, loss_dict_jieba = train_model(datasets['jieba'], 'Jieba')
    models['jieba'] = model_jieba
    vocabs['jieba'] = vocab_jieba 
    losses['jieba'] = loss_dict_jieba['epoch']
    step_losses['jieba'] = loss_dict_jieba['step']
    # 训练SPM模型 
    model_spm, vocab_spm, loss_dict_spm = train_model(datasets['spm'], 'SPM')
    models['spm'] = model_spm
    vocabs['spm'] = vocab_spm
    losses['spm'] = loss_dict_spm['epoch']
    step_losses['spm'] = loss_dict_spm['step']
    
    # 可视化训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses['jieba'], label='Jieba')
    plt.plot(losses['spm'], label='SPM')
    plt.title('训练损失对比')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.savefig('training_comparison.png')
    
    # 模型对比
    compare_models(models, vocabs)
    
    # 保存模型
    torch.save(models['jieba'].state_dict(), 'comments_jieba.pth')
    torch.save(models['spm'].state_dict(), 'comments_spm.pth')

