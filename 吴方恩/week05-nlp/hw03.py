import fasttext
import torch

from torch.utils.tensorboard import SummaryWriter

model = fasttext.train_supervised(input="吴方恩/week05-nlp/resources/cooking.train",epoch=25,lr=1.0)

# model.save_model("吴方恩/week05-nlp/resources/train_supervised_model.bin")

    
#  tensorboard 可视化
writer = SummaryWriter("吴方恩/week05-nlp/runs/cooking")

metadata = list(set(model.words))
embeddings = []

for word in metadata:
    embeddings.append(model.get_word_vector(word))
    
print(f"元数据数量: {len(metadata)}")  # 调试输出
print(f"embeddings数量: {len(embeddings)}")  # 调试输出
writer.add_embedding(torch.tensor(embeddings),metadata=metadata)

writer.close()

print(model.predict("Which baking dish is best to bake a banana bread ?"))

print(model.predict("Why not put knives in the dishwasher?"))

print(model.test("吴方恩/week05-nlp/resources/cooking.valid"))