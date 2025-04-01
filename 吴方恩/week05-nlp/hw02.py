import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import torch

from torch.utils.tensorboard import SummaryWriter


# 预处理分词文件
def preprocess_file():
    # 读取修复后的评论文件
    lines = [line for line in open('吴方恩/week05-nlp/resources/三体.txt')]
    with open('吴方恩/week05-nlp/resources/三体.txt','w') as f:
        for line in lines:
            terms = jieba.lcut(line)
            f.write(' '.join(terms)+'\n')

# preprocess_file()
model = fasttext.train_unsupervised('吴方恩/week05-nlp/resources/三体.txt', model='skipgram')



def cosine_word(word1,word2):
    vector1 = model.get_word_vector(word1).reshape(1, -1)
    vector2 = model.get_word_vector(word2).reshape(1, -1)
    similarity = cosine_similarity(vector1,vector2)
    return similarity[0][0]
print(f"三体和云天明的相似度:{cosine_word('三体','云天明')*100:.2f}%")
print(f"三体和程心的相似度:{cosine_word('三体','程心')*100:.2f}%")
print(f"三体和罗辑的相似度:{cosine_word('三体','罗辑')*100:.2f}%")
print(f"三体和叶文洁的相似度:{cosine_word('三体','叶文洁')*100:.2f}%")
# 查找与目标最相似的词
target_word = "叶文洁"
similar_words = model.get_nearest_neighbors(target_word, k=3)
print(f"与 '{target_word}' 最相似的词及相似度:")
for score, word in similar_words:
    print(f"{word}: {score:.4f}")


#  tensorboard 可视化
writer = SummaryWriter("吴方恩/week05-nlp/runs/word2vec")

metadata = list(set(model.words))
embeddings = []

for word in metadata:
    embeddings.append(model.get_word_vector(word))
    
print(f"元数据数量: {len(metadata)}")  # 调试输出
print(f"embeddings数量: {len(embeddings)}")  # 调试输出
writer.add_embedding(torch.tensor(embeddings),metadata=metadata)

writer.close()

