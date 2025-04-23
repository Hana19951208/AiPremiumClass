import csv
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import json
import random
from tqdm import tqdm
import os


# 设置随机种子
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 数据预处理
def load_data(filename):
    comments, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                star = int(row['Star'])
                comment = row['Comment'].strip()
                if star in (1, 5):
                    labels.append(0 if star == 1 else 1)
                    comments.append(comment)
            except (KeyError, ValueError):
                continue
    return comments, labels

# 中文分词处理
def process_text(text, min_len=10, max_len=100):
    try:
        words = list(jieba.cut(text))
        return words if min_len <= len(words) <= max_len else None
    except:
        return None

# 构建并保存词典
def build_and_save_vocab(processed_texts, vocab_file='/kaggle/working/vocab.json', vocab_size=20000):
    word_counter = Counter()
    for words in processed_texts:
        word_counter.update(words)
    
    vocab = ['<PAD>', '<UNK>'] + [w for w, _ in word_counter.most_common(vocab_size)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False)
    
    return word2idx

# 加载词典
def load_vocab(vocab_file='/kaggle/working/vocab.json'):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# 转换索引序列
def text_to_indices(words, word2idx):
    return [word2idx.get(word, 1) for word in words]  # 1表示UNK

# 数据集类
class TextDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.indices[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

# 动态填充函数
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    sorted_indices = torch.argsort(lengths, descending=True)
    
    sorted_texts = [texts[i] for i in sorted_indices]
    padded = torch.nn.utils.rnn.pad_sequence(
        sorted_texts, batch_first=True, padding_value=0
    )
    return padded, torch.tensor([labels[i] for i in sorted_indices]), lengths[sorted_indices]

# RNN模型
class DynamicRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (hidden, _) = self.rnn(packed)
        return self.fc(hidden[-1])

# 主流程
if __name__ == "__main__":
    # 1. 数据加载与预处理
    comments, labels = load_data('/kaggle/input/doubanmovieshortcomments/DMSC.csv')
    
    # 2. 分词过滤
    processed = []
    valid_indices = []
    for i, text in enumerate(comments):
        words = process_text(text)
        if words:
            processed.append(words)
            valid_indices.append(i)
    labels = [labels[i] for i in valid_indices]
    
    # 3. 构建并保存词典
    word2idx = build_and_save_vocab(processed)
    vocab_size = len(word2idx)
    
    # 4. 转换索引
    indices = [text_to_indices(words, word2idx) for words in processed]
    
    # 5. 创建数据集
    dataset = TextDataset(indices, labels)
    
    # 6. 划分数据集
    train_size = int(len(dataset)*0.8)
    train_set, test_set = random_split(dataset, [train_size, len(dataset)-train_size])
    
    # 7. 创建DataLoader
    BATCH_SIZE = 128
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                           collate_fn=collate_fn)
    
    # 8. 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DynamicRNN(vocab_size, 128, 256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 9. 训练循环（添加进度条）
    best_acc = 0.0
    for epoch in range(10):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for inputs, labels, lengths in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{total_loss/(progress_bar.n+1):.3f}"})
        
        # 验证并保存最佳模型
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels, lengths in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, lengths)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
        
        current_acc = correct / len(test_set)
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': 128,
                'hidden_dim': 256
            }, '/kaggle/working/best_model.pth')
        
        print(f'Epoch {epoch+1}, Loss: {total_loss:.3f}, Acc: {current_acc:.3f}')
   