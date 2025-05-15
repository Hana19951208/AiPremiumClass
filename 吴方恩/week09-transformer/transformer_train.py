import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

# 位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # 行缩放指数值
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 位置编码索引 (5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 编码矩阵 (5000, emb_size)
        pos_embdding = torch.zeros((maxlen, emb_size))
        pos_embdding[:, 0::2] = torch.sin(pos * den)
        pos_embdding[:, 1::2] = torch.cos(pos * den)
        # 添加和batch对应维度 (1, 5000, emb_size)
        pos_embdding = pos_embdding.unsqueeze(0)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        token_len = token_embdding.size(1)  # token长度
        # (1, token_len, emb_size)
        add_emb = self.pos_embedding[:, :token_len] + token_embdding
        return self.dropout(add_emb)

class Seq2SeqTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        # transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        # encoder input embedding
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        # decoder input embedding
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask, 
                         tgt_key_padding_mask=dec_pad_mask)
        # 推理
        return self.predict(outs)
    
    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)
    
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)
    
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
    
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    # 构建encoder和docoder的词典

    # 模型训练数据： X：([enc_token_matrix], [dec_token_matrix] shifted right)，
    # y [dec_token_matrix] shifted
    # 根据第一个字生成后面的字,例如: 人->生得意须尽欢，莫使金樽空对月  人生->得意须尽欢，莫使金樽空对月

    # 1. 通过词典把token转换为token_index
    # 2. 通过Dataloader把encoder，decoder封装为带有batch的训练数据
    # 3. Dataloader的collate_fn调用自定义转换方法，填充模型训练数据
    #    3.1 encoder矩阵使用pad_sequence填充
    #    3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
    #    3.3 decoder后面部分训练目标 dec_token_matrix[:,1:,:]
    # 4. 创建mask
    #    4.1 dec_mask 上三角填充-inf的mask
    #    4.2 enc_pad_mask: (enc矩阵 == 0）
    #    4.3 dec_pad_mask: (dec矩阵 == 0)
    # 5. 创建模型（根据GPU内存大小设计编码和解码器参数和层数）、优化器、损失
    # 6. 训练模型并保存

    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import Dataset
    import os

    # 1. 构建词典
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    special_tokens = ['<pad>', '<s>', '</s>']
    vocab = special_tokens + list(set(chs))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)


    # 2. 构建 Dataset
    class TextDataset(Dataset):
        def __init__(self, enc_tokens, dec_tokens):
            self.enc_data = [encode_tokens(tokens) for tokens in enc_tokens]
            self.dec_data = [encode_tokens(tokens) for tokens in dec_tokens]

        def __len__(self):
            return len(self.enc_data)

        def __getitem__(self, idx):
            return self.enc_data[idx], self.dec_data[idx]

    def collate_fn(batch):
        enc_batch, dec_batch = zip(*batch)
        # print(enc_batch, dec_batch)
        enc_batch = pad_sequence(enc_batch, batch_first=True, padding_value=PAD_IDX)
        dec_batch = pad_sequence(dec_batch, batch_first=True, padding_value=PAD_IDX)

        # decoder input: 全部减去最后一个token
        dec_input = dec_batch[:, :-1]
        # decoder target: 全部去掉第一个token
        dec_target = dec_batch[:, 1:]
        return enc_batch, dec_input, dec_target

    dataset = TextDataset(enc_tokens, dec_tokens)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

    # 3. Mask 函数
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
        return mask

    # 4. 构建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model=32, nhead=2, num_enc_layers=2, num_dec_layers=2,
                            dim_forward=128, dropout=0.1,
                            enc_voc_size=vocab_size, dec_voc_size=vocab_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 5. 开始训练
    for epoch in range(500):
        model.train()
        total_loss = 0
        for enc_batch, dec_input, dec_target in dataloader:
            enc_batch, dec_input, dec_target = enc_batch.to(device), dec_input.to(device), dec_target.to(device)
            
            tgt_mask = generate_square_subsequent_mask(dec_input.size(1)).to(device)
            enc_pad_mask = enc_batch == PAD_IDX
            dec_pad_mask = dec_input == PAD_IDX

            output = model(enc_batch, dec_input, tgt_mask, enc_pad_mask, dec_pad_mask)
            output = output.reshape(-1, output.shape[-1])
            dec_target = dec_target.reshape(-1)

            loss = criterion(output, dec_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("模型已保存为 transformer_model.pth")
