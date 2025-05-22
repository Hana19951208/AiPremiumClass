import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModelForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

MODEL_PATH = '/kaggle/working/'
PRETRAINED_MODEL = 'bert-base-chinese'
LOG_DIR = './runs/bert_cls'
BATCH_SIZE = 256
EPOCHS = 5
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集定义
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 加载并预处理数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row['评价内容(content)'] not in ['此用户未填写评价内容', '您没有填写内容，默认好评'] and row['评价内容(content)'].strip() != '']
    texts = [row['评价内容(content)'] for row in rows]
    labels = [int(row['评分（总分5分）(score)']) - 1 for row in rows]  # 标签从0开始
    split_idx = int(len(texts) * 0.8)
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    test_texts, test_labels = texts[split_idx:], labels[split_idx:]
    return train_texts, train_labels, test_texts, test_labels

# 冻结 BERT 编码层
def freeze_bert(model):
    for param in model.bert.parameters():
        param.requires_grad = False

# 验证函数
def evaluate(model, data_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    model.train()
    return correct / total

# 主训练函数
def train(freeze=False):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    train_texts, train_labels, test_texts, test_labels = load_data('jd_comment_data.csv')

    train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = CommentDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=5)

    if freeze:
        freeze_bert(model)

    model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=LOG_DIR + ('/freeze' if freeze else '/finetune'))

    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        torch.save(model.module.state_dict(), os.path.join(MODEL_PATH, f"{'freeze' if freeze else 'finetune'}_bert_epoch{epoch+1}.pt"))
        # 在每个 epoch 后进行验证
        val_acc = evaluate(model, test_loader)
        writer.add_scalar("eval/accuracy", val_acc, epoch)
        print(f"Epoch {'freeze' if freeze else 'finetune'} {epoch+1} completed. Avg loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}")

    writer.close()


if __name__ == '__main__':
    # 冻结 BERT 主体
    train(freeze=True)   # 冻结只训练分类头
    train(freeze=False)  # 不冻结，完整微调
