import pickle
import torch
from torch.utils.data import DataLoader
from process import get_proc
from EncoderDecoderModel import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

if __name__ == '__main__':
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir,'data')
    enc_dec_data_file = os.path.join(base_dir, 'enc_dec_data.bin')
    enc_dec_test_data = os.path.join(base_dir, 'enc_dec_test_data.bin')
    vocab_file = os.path.join(base_dir, 'vocab.bin')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据
    with open(vocab_file,'rb') as f:
        evoc,dvoc = pickle.load(f)

    with open(enc_dec_data_file,'rb') as f:
        enc_data,dec_data = pickle.load(f)
    with open(enc_dec_test_data,'rb') as f:
        enc_test_data,dec_test_data = pickle.load(f)

    ds = list(zip(enc_data,dec_data))
    ts = list(zip(enc_test_data,dec_test_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=get_proc(evoc, dvoc))
    tl = DataLoader(ts, batch_size=256, shuffle=False, collate_fn=get_proc(evoc, dvoc))

    # 构建训练模型
    # 模型构建
    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120
        )
    model.to(device)

    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 训练
    for epoch in range(5):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dl, desc=f'Epoch {epoch+1}')
        for enc_input, dec_input, targets, enc_lengths in progress_bar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            # 前向传播 
            logits, _ = model(enc_input, dec_input, enc_lengths)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({'loss': f"{total_loss/(progress_bar.n+1):.3f}"})
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            test_progress = tqdm(tl, desc=f'Validating Epoch {epoch+1}', leave=False)
            for enc_input, dec_input, targets in test_progress:
                enc_input = enc_input.to(device)
                dec_input = dec_input.to(device)
                targets = targets.to(device)

                logits, _ = model(enc_input, dec_input)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                test_loss += loss.item()

                # 计算准确率
                preds = logits.argmax(dim=-1)
                mask = targets != 0  # 假设0是padding的索引
                batch_correct = (preds[mask] == targets[mask]).sum().item()
                batch_total = mask.sum().item()
                
                total_correct += batch_correct
                total_tokens += batch_total
                
                # 更新进度条
                test_progress.set_postfix({
                    'val_loss': f"{test_loss/(test_progress.n+1):.3f}",
                    'val_acc': f"{(total_correct/total_tokens) if total_tokens!=0 else 0:.3f}"
                })

        # 计算验证指标
        avg_test_loss = test_loss / len(tl)
        test_accuracy = total_correct / total_tokens if total_tokens != 0 else 0
        
        # 打印验证结果
        print(f"\nEpoch {epoch+1} Validation - Loss: {avg_test_loss:.3f} | Accuracy: {test_accuracy:.3f}")

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(base_dir, 'best_seq2seq_state.bin')
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved with accuracy: {best_accuracy:.3f}\n")


    torch.save(model.state_dict(), os.path.join(base_dir, 'seq2seq_state.bin'))