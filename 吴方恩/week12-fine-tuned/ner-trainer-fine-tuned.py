# %%writefile ner_ddp.py

import os
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
import evaluate
from datasets import load_dataset
import torch.distributed as dist
import torch.multiprocessing as mp


# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    # 1. 分词器
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    
    # 2. 加载新的数据集
    ds_raw = load_dataset('doushabao4766/msra_ner_k_V3')
    
    # 3. 定义 MSRA NER 的实体和标签
    msra_entities = ['O', 'PER', 'ORG', 'LOC']
    tags = ['O']
    for entity in msra_entities[1:]:
        tags.append('B-' + entity)
        tags.append('I-' + entity)
    
    id2lbl = {i: tag for i, tag in enumerate(tags)}
    lbl2id = {tag: i for i, tag in enumerate(tags)}
    num_labels = len(tags)
    
    # 4. 过滤掉 ner_tags 全为零的样本
    def filter_no_entities(item):
        return not all(tag == 0 for tag in item['ner_tags'])
    
    print(f"原始数据集大小: { {split: len(ds_raw[split]) for split in ds_raw.keys()} }")
    ds_filtered = ds_raw.filter(filter_no_entities)
    print(f"过滤后数据集大小: { {split: len(ds_filtered[split]) for split in ds_filtered.keys()} }")
    
    # 5. 数据预处理函数
    def data_input_proc(item):
        tokenized_inputs = tokenizer(
            item['tokens'],
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=False, # 保持False，因为我们直接使用预分词的tokens
            max_length=512
        )
        labels_batch = []
        for i, label_list in enumerate(item['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_list[word_idx])
                else:
                    label_ids.append(-100) # 或 label_list[word_idx] 如果希望标记所有子词元
                previous_word_idx = word_idx
            labels_batch.append(label_ids)
        tokenized_inputs["labels"] = labels_batch
        return tokenized_inputs
    
    ds_processed = ds_filtered.map(data_input_proc, batched=True, remove_columns=ds_filtered['train'].column_names)
    ds_processed.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    
    # 6. 训练参数
    local_rank = rank
    args = TrainingArguments(
        output_dir="msra_ner_train_output_cn_extraction",
        num_train_epochs = 2,    # 训练 epoch
        save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
        per_device_train_batch_size=16,  # 训练批次
        per_device_eval_batch_size=16,
        report_to='tensorboard',  # 训练输出记录
        eval_strategy="epoch",
        local_rank=local_rank,   # 当前进程 RANK
        fp16=True,               # 使用混合精度
        lr_scheduler_type='linear',  # 动态学习率
        learning_rate=2e-5,
        warmup_steps=100,        # 预热步数
        ddp_find_unused_parameters=False  # 优化DDP性能
    )
    
    # 7. 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        num_labels=num_labels,
        id2label=id2lbl,
        label2id=lbl2id
    )
    model.to(local_rank)

    # 8. 指标计算函数
    seqeval_metric = evaluate.load('seqeval')
    def compute_metrics(p):
        predictions, labels = p.predictions, p.label_ids
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [tags[pred_idx] for (pred_idx, lbl_idx) in zip(prediction, label) if lbl_idx != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [tags[lbl_idx] for (pred_idx, lbl_idx) in zip(prediction, label) if lbl_idx != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    # 9. 数据整理器
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    # 10. Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=ds_processed['train'],
        eval_dataset=ds_processed['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # 评估集处理
    eval_dataset_key = 'test'
    print(f"使用 '{eval_dataset_key}' 集进行评估。")
    trainer.eval_dataset = ds_processed[eval_dataset_key]
    
    
    # 11. 训练
    print("开始训练...")
    trainer.train()
    # --- 新增实体抽取函数 ---
    def extract_entities_from_text(text, model, tokenizer, id2label_map):
        # 将模型置于评估模式
        model.eval()
        # 将模型和输入数据移动到同一设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
        # 1. 分词并准备模型输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()} # 将输入移动到设备
    
        # 2. 模型预测
        with torch.no_grad(): # 关闭梯度计算
            outputs = model(**inputs)
    
        # 3. 获取预测结果
        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_label_ids = predictions[0].cpu().tolist() # 取第一个样本的结果，并转到CPU转为list
        input_ids = inputs["input_ids"][0].cpu().tolist()
    
        # 4. 将标签ID转换为标签字符串，并与原始词元对齐
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        predicted_tags = []
        # -100是tokenizer的pad_token_id的默认标签，bert-base-chinese的pad_token_id是0
        # 我们只关心非特殊标记（非[CLS], [SEP], [PAD]）的预测
        # CLS id: 101, SEP id: 102, PAD id: 0 (for bert-base-chinese)
        special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    
        meaningful_tokens = []
        meaningful_tags = []
    
        for token_id, label_id, token_str in zip(input_ids, predicted_label_ids, tokens):
            if token_id not in special_token_ids:
                meaningful_tokens.append(token_str)
                meaningful_tags.append(id2label_map[label_id])
            if token_id == tokenizer.sep_token_id: # 到达SEP后停止
                break
                
        # 5. 聚合实体
        entities = []
        current_entity_content = []
        current_entity_label = None
    
        for token_char, tag_str in zip(meaningful_tokens, meaningful_tags):
            # bert-base-chinese 对于中文是字符级，token_char 就是单个汉字
            # 对于英文等可能会有 "##" 前缀的子词，这里简单处理，假设中文场景
            # 如果是子词，可能需要更复杂的聚合逻辑，但bert-base-chinese对中文是字符级
            
            if tag_str.startswith("B-"):
                # 如果当前有正在构建的实体，先保存它
                if current_entity_content:
                    entities.append({
                        "entity": current_entity_label,
                        "content": "".join(current_entity_content)
                    })
                # 开始新的实体
                current_entity_label = tag_str[2:] # 例如 "ORG"
                current_entity_content = [token_char]
            elif tag_str.startswith("I-"):
                # 如果I标签与当前实体类型匹配，并且确实有一个当前实体在构建中
                if current_entity_label and tag_str[2:] == current_entity_label:
                    current_entity_content.append(token_char)
                else:
                    # I标签与B标签不匹配，或者没有B标签就开始了I标签
                    # 此时，如果之前有实体，保存它
                    if current_entity_content:
                        entities.append({
                            "entity": current_entity_label,
                            "content": "".join(current_entity_content)
                        })
                    # 将这个I标签视为一个新的B标签开始 (一种容错处理)
                    current_entity_label = tag_str[2:]
                    current_entity_content = [token_char]
            else: # O标签或其他 (理论上应该是O)
                # 如果当前有正在构建的实体，保存它
                if current_entity_content:
                    entities.append({
                        "entity": current_entity_label,
                        "content": "".join(current_entity_content)
                    })
                # 重置当前实体状态
                current_entity_content = []
                current_entity_label = None
                
        # 循环结束后，如果仍有未保存的实体
        if current_entity_content:
            entities.append({
                "entity": current_entity_label,
                "content": "".join(current_entity_content)
            })
            
        return entities
    
    # --- 训练和评估完成后，使用新函数进行预测 ---
    print("\n--- 开始使用训练好的模型进行实体抽取 ---")
    
    # 确保模型和分词器已加载并训练/微调完毕
    # trainer.model 就是训练好的模型
    trained_model = trainer.model 
    
    test_sentence_1 = "双方确定了今后发展中美关系的指导方针。"
    extracted_entities_1 = extract_entities_from_text(test_sentence_1, trained_model, tokenizer, id2lbl)
    print(f"输入: \"{test_sentence_1}\"")
    print(f"输出: {extracted_entities_1}")
    
    print("\n训练和预测完成。")
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()