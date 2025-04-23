import torch
import os
from hw03 import process_text, load_vocab, DynamicRNN


# 预测函数（支持加载保存的模型）
def predict(text, model_path='best_model.pth', vocab_file='vocab.json'):
    # 加载词典
    word2idx = load_vocab(vocab_file)
    # 转换为int型键
    word2idx = {k: int(v) for k, v in word2idx.items()}
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = DynamicRNN(
        checkpoint['vocab_size'],
        checkpoint['embed_dim'],
        checkpoint['hidden_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 预处理
    words = process_text(text, min_len=1, max_len=20)
    if not words: return None
    indices = [word2idx.get(word, 1) for word in words]
    
    # 转换为Tensor
    tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    length = torch.tensor([len(indices)]).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(tensor, length)
    return '好评' if torch.argmax(output).item() == 1 else '差评'

if __name__ == "__main__":
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'data/best_model.pth')
    voca_file = os.path.join(script_dir, 'data/vocab.json')
    # 测试样例
    test_samples = [
        "我真服了，这个电影太烂了，我不喜欢看了",
        "老戏骨真帅，剧情也不错，推荐给大家",
        "还是算了吧，小鲜肉的演技太烂了"
    ]
    for text in test_samples:
        print(f'评论：「{text}」\n预测结果：{predict(text,model_path=model_path,vocab_file=voca_file)}\n')