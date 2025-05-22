import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = './saved_model/bert_epoch5.pt'
PRETRAINED_MODEL = 'bert-base-chinese'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 tokenizer 和 模型
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=5)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(texts):
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        preds = torch.argmax(outputs.logits, dim=1)
    return preds.cpu().numpy().tolist()

if __name__ == '__main__':
    test_texts = [
        "这款手机性价比很高，值得购买",
        "包装很差，收到的时候已经坏了",
        "物流挺快，客服也不错",
        "一般般，没有描述的那么好",
        "太棒了，非常满意的一次购物"
    ]
    results = predict(test_texts)
    for text, label in zip(test_texts, results):
        print(f"评论: {text} -> 预测评分: {label+1} 星")
