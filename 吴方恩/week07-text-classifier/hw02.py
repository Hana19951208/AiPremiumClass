# 加载处理后文本构建词典、定义模型、训练、评估、测试
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import  os
import jieba
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度

# 获取文件当前路径
current_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_path, "data/comments.pkl")

with open(file_path, 'rb') as f:
    ds_comments = pickle.load(f)



def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

# 定义embedding
class DocEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(DocEmbedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_ids):
        return self.embedding_matrix[input_ids]
# 数据转换
def convert_data(batch_data):
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

# 定义模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, (hidden, _) = self.lstm(embedded)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output
# 定义超参数
embedding_dim = 200
hidden_dim = 256
output_dim = 2
learning_rate = 0.001
num_epochs = 20
batch_size = 64

model_path = os.path.join(current_path, "data/comments_classifier.pth")
vocab_path = os.path.join(current_path, "data/comments_vocab.pth")
# 主函数
if __name__ == '__main__':
    
    # 查看数据好评数和差评数，并查看其在数据中的占比
    positive_count = 0
    negative_count = 0
    for comment, vote in ds_comments:
        if vote == 1:
            positive_count += 1
        else:
            negative_count += 1
    print('好评数:', positive_count)
    print('差评数:', negative_count)
    print('好评占比:', positive_count / (positive_count + negative_count))
    print('差评占比:', negative_count / (positive_count + negative_count))
    
    vocab = build_from_doc(ds_comments)
    print('词汇表大小:', len(vocab))
    vocab_size = len(vocab)
    dataloader = DataLoader(ds_comments, batch_size=batch_size, shuffle=True, collate_fn=convert_data)
    # 初始化模型、损失函数和优化器
    model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # 添加平滑因子防止除零
    epsilon = 1e-5
    # 在训练循环前添加类别权重计算
    class_weights = torch.tensor([(positive_count + epsilon) / (negative_count + epsilon),1.0], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(num_epochs):
        for i, (batch_comments, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch_comments)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')


    # 保存模型
    torch.save(model.state_dict(), model_path)
    # 模型词典
    torch.save(vocab, vocab_path)

    """"""""""""""""""""""""""""""""""""""""""""""""""


    # 加载词典
    vocab = torch.load(vocab_path)
    # 测试模型
    comment1 = '这部电影真好看！全程无尿点'
    comment2 = '看到一半就不想看了，太无聊了，演员演技也很差'

    # 将评论转换为索引
    comment1_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment1)])
    comment2_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment2)])
    # 将评论转换为tensor
    comment1_idx = comment1_idx.unsqueeze(0) # 添加batch维度    
    comment2_idx = comment2_idx.unsqueeze(0) # 添加batch维度

    # 加载模型
    model = TextClassifier(len(vocab), embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))

    # 模型推理
    pred1 = model(comment1_idx)
    pred2 = model(comment2_idx)

    # 取最大值的索引作为预测结果
    pred1 = torch.argmax(pred1, dim=1).item()
    pred2 = torch.argmax(pred2, dim=1).item()
    print(f'评论1预测结果: {pred1}')
    print(f'评论2预测结果: {pred2}')
