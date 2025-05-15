from transformer_train import Seq2SeqTransformer,generate_square_subsequent_mask
import torch


def greedy_decode(model, src_sentence, max_len=20):
    model.eval()
    src = encode_tokens(src_sentence)
    src = src.unsqueeze(0).to(device)  # batch 维度

    enc_pad_mask = src == PAD_IDX
    memory = model.encode(src)

    ys = torch.tensor([[word2idx['<s>']]], dtype=torch.long).to(device)  # 解码起点

    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = model.predict(out)[:, -1, :]  # 取最后一个 token 的 logits
        next_token = out.argmax(-1).item()

        if next_token == EOS_IDX or next_token == PAD_IDX:
            break

        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)

    return [idx2word[token.item()] for token in ys[0][1:]]  # 去掉<s>，只返回生成部分

def encode_tokens(tokens):
    return torch.tensor([word2idx[token] for token in tokens], dtype=torch.long)

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chs = list("人生得意须尽欢，莫使金樽空对月")
    # 构建词典
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    special_tokens = ['<pad>', '<s>', '</s>']
    vocab = special_tokens + list(set(chs))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)

    # 加载模型
    model = Seq2SeqTransformer(d_model=32, nhead=2, num_enc_layers=2, num_dec_layers=2,
                            dim_forward=128, dropout=0.1,
                            enc_voc_size=vocab_size, dec_voc_size=vocab_size).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load("transformer_model.pth", map_location=device))

    # 推理：输入开头“人生”
    start_text = ['人', '生','得','意']
    output = greedy_decode(model, start_text)
    print("生成结果：", ''.join(output))
