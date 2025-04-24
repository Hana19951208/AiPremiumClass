import torch
import pickle
from EncoderDecoderModel import Encoder, Decoder, Seq2Seq
import os

class CoupletGenerator:
    def __init__(self, model_path, vocab_path, device='cpu'):
        self.device = device
        with open(vocab_path, 'rb') as f:
            self.enc_vocab, self.dec_vocab = pickle.load(f)
        self.idx2word = {v:k for k,v in self.dec_vocab.items()}
        
        # 初始化模型
        encoder = Encoder(len(self.enc_vocab), 256, 512)
        decoder = Decoder(len(self.dec_vocab), 256, 512)
        self.model = Seq2Seq(encoder, decoder, device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def generate(self, input_str, max_len=50):
        # 预处理输入
        input_seq = [self.enc_vocab.get(c, self.enc_vocab['<UNK>']) for c in input_str]
        input_tensor = torch.LongTensor([input_seq]).to(self.device)
        length_tensor = torch.LongTensor([len(input_seq)]).to(self.device)
        
        with torch.no_grad():
            enc_out, hidden = self.model.encoder(input_tensor, length_tensor)
            
        # 生成序列
        output = [self.dec_vocab['BOS']]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([output[-1:]]).to(self.device)
            dec_out, hidden = self.model.decoder(trg_tensor, hidden, enc_out)
            next_word = dec_out.argmax(-1)[-1].item()
            if next_word == self.dec_vocab['EOS']:
                break
            output.append(next_word)
        
        return ''.join([self.idx2word.get(i, '?') for i in output[1:]])

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    generator = CoupletGenerator(
        model_path=os.path.join(base_dir, 'seq2seq-state.bin'),
        vocab_path=os.path.join(base_dir, 'vocab.bin'),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    test_input = "春风得意马蹄疾"
    output = generator.generate(test_input)
    print(f"上联：{test_input}")
    print(f"生成下联：{output}")