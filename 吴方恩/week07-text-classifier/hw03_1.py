import csv
import jieba
import os
import pickle
import sentencepiece as spm
import tempfile

# 定义文件路径
script_dir = '/Users/Hana/Downloads/八斗学院/AiPremiumClass/吴方恩/week07-text-classifier'
input_file = os.path.join(script_dir, "data/DMSC-CUG King of Heroes.csv")
spm_comms_file = os.path.join(script_dir, "data/spm_comms.txt")

# 初始化数据容器
raw_comments_with_votes = []  # 存储原始评论和标签
jieba_comments = []          # 存储jieba分词结果
spm_comments = []            # 存储sentencepiece分词结果

# 第一步：读取原始数据并存储
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)  # 使用DictReader读取列名
    
    pos_count = 0
    neg_count = 0
    
    for row in reader:
        try:
            # 获取投票标签
            vote = 1 if int(row['Star']) > 3 else 0
            
            # 控制样本数量
            if vote == 1 and pos_count >= 50000:
                continue
            if vote == 0 and neg_count >= 50000:
                continue
                
            # 记录原始数据
            raw_comment = row['Comment']
            raw_comments_with_votes.append((raw_comment, vote))
            
            # 更新计数器
            if vote == 1:
                pos_count += 1
            else:
                neg_count += 1
                
            # 提前终止条件
            if pos_count >= 50000 and neg_count >= 50000:
                break
                
        except Exception as e:
            print(f"处理行时出错: {row}，错误: {e}")

# 第二步：生成jieba分词结果
for comment, vote in raw_comments_with_votes:
    jieba_tokens = jieba.lcut(comment)
    jieba_comments.append((jieba_tokens, vote))

# 第三步：生成sentencepiece分词结果
with open(spm_comms_file,'w',encoding='utf-8') as temp_f:
    # 写入临时文件供sentencepiece训练
    for comment, _ in raw_comments_with_votes:
        temp_f.write(comment + '\n')

# 训练sentencepiece模型
# spm.SentencePieceTrainer.Train(
#     input=spm_comms_file,
#     model_prefix='spm_model',
#     vocab_size=5000,
#     user_defined_symbols=['<pad>', '<unk>'],
#     pad_id=0,
#     unk_id=1
# )
spm.SentencePieceTrainer.Train(input=spm_comms_file, 
                               model_prefix='spm_model',
                               vocab_size=10000) # Train the model

# 加载模型
sp = spm.SentencePieceProcessor()
sp.load('spm_model.model')

# 进行分词
for comment, vote in raw_comments_with_votes:
    sp_tokens = sp.encode_as_pieces(comment)
    spm_comments.append((sp_tokens, vote))

# 第四步：保存结果
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# 保存词长度为10-120的样本
jieba_comments = [c for c in jieba_comments if len(c[0]) in range(10, 120)]
spm_comments = [c for c in spm_comments if len(c[0]) in range(10, 120)]

# 保存jieba分词结果
with open(os.path.join(data_dir, 'comments_jieba.pkl'), 'wb') as f:
    pickle.dump(jieba_comments, f)

# 保存sentencepiece分词结果 
with open(os.path.join(data_dir, 'comments_spm.pkl'), 'wb') as f:
    pickle.dump(spm_comments, f)

# 清理临时文件
os.remove('spm_model.model')
os.remove('spm_model.vocab')

print(f"处理完成，Jieba样本数: {len(jieba_comments)}, SPM样本数: {len(spm_comments)}")
