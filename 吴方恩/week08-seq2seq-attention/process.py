import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import json
import pickle

# 读取数据
def read_file(enc_file, dec_file):
    
    enc_data, dec_data = [], []
    with open(enc_file, 'r', encoding='utf-8') as f:
        enc_lines = f.read().split('\n')
        
        for line in enc_lines:
            if line == '':
                continue
            line = line.replace('，','').replace('。','').replace('！','').replace('？','').replace(' ','')
            enc_data.append(list(line))
            
    with open(dec_file, 'r', encoding='utf-8') as f:
        dec_lines = f.read().split('\n')

        for line in dec_lines:
            if line == '':
                continue
            line = line.replace('，','').replace('。','').replace('！','').replace('？','').replace(' ','')
            dec_tks = ['BOS'] + list(line) + ['EOS']
            dec_data.append(dec_tks)
    assert len(enc_data) == len(dec_data), '编码数据与解码数据长度不一致！'
    return enc_data, dec_data

# 构建字典
def bulid_vocab(data):
    vocab = set()
    for line in data:
        vocab.update(line)
    vocab = sorted(list(vocab))
    vocab = ['<PAD>','<UNK>'] + vocab
    return {tk:idx for idx, tk in enumerate(vocab)}

def get_proc(enc_voc, dec_voc):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        enc_ids, dec_ids, labels, enc_lengths = [], [], [], []
        for enc, dec in data:
            enc_idx = [enc_voc.get(tk, enc_voc['<UNK>']) for tk in enc]
            dec_idx = [dec_voc.get(tk, dec_voc['<UNK>']) for tk in dec]
            
            enc_ids.append(torch.tensor(enc_idx))
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            labels.append(torch.tensor(dec_idx[1:]))
            enc_lengths.append(len(enc_idx))

        # 按enc长度降序排列
        sorted_indices = sorted(range(len(enc_lengths)), key=lambda i: enc_lengths[i], reverse=True)
        enc_ids = [enc_ids[i] for i in sorted_indices]
        dec_ids = [dec_ids[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        enc_lengths = [enc_lengths[i] for i in sorted_indices]

        # 填充
        enc_input = pad_sequence(enc_ids, batch_first=True, padding_value=0)
        dec_input = pad_sequence(dec_ids, batch_first=True, padding_value=0)
        targets = pad_sequence(labels, batch_first=True, padding_value=0)
        return enc_input, dec_input, targets, enc_lengths

    # 返回回调函数
    return batch_proc    

if __name__ == '__main__':
    



    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir,'data')
    enc_file = os.path.join(base_dir, 'train/in.txt')
    dec_file = os.path.join(base_dir, 'train/out.txt')
    # 读取数据
    enc_data,dec_data = read_file(enc_file, dec_file)
    
    # 保存数据
    enc_dec_data_file = os.path.join(base_dir, 'enc_dec_data.bin')
    with open(enc_dec_data_file, 'wb') as f:
        pickle.dump((enc_data, dec_data),f)
        
    print('enc length', len(enc_data))
    print('dec length', len(dec_data))
    print(enc_data[:2])
    print(dec_data[:2])
    

    
    enc_test_file = os.path.join(base_dir, 'test/in.txt')
    dec_test_file = os.path.join(base_dir, 'test/out.txt')
    # 读取数据
    enc_test_data,dec_test_data = read_file(enc_test_file, dec_test_file)
    
    # 保存数据
    enc_dec_test_data_file = os.path.join(base_dir, 'enc_dec_test_data.bin')
    with open(enc_dec_test_data_file, 'wb') as f:
        pickle.dump((enc_test_data, dec_test_data),f)

    # 构建字典
    enc_voc = bulid_vocab(enc_data+enc_test_data)
    dec_voc = bulid_vocab(dec_data+dec_test_data)
    
    vocab_file = os.path.join(base_dir, 'vocab.bin')

    # 保存字典
    with open(vocab_file,'wb') as f:
        pickle.dump((enc_voc, dec_voc),f)
    print('enc vocab size', len(enc_voc))   
    print('dec vocab size', len(dec_voc))

    dataset = list(zip(enc_data, dec_data))
    
    # 数据加载器
    batch_size = 2
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=get_proc(enc_voc, dec_voc))
    for enc_input, dec_input, targets,enc_lengths in train_loader:
        print('enc_input shape', enc_input.shape)
        print('dec_input shape', dec_input.shape)
        print('targets shape', targets.shape)
        break
    


    