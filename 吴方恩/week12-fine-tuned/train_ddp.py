#  %%writefile train_ddp.py

import os
import torch
import numpy as np
import evaluate
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)

# --- 1. 分布式环境设置 ---
def setup(rank, world_size):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 'nccl' 是 NVIDIA GPU 推荐的后端
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

# --- 2. 核心训练函数 ---
def train_process(rank, world_size):
    """
    该函数将在每个GPU上独立运行。
    `rank` 是当前进程的ID(从0到world_size-1)。
    `world_size` 是总的进程数(通常等于GPU数量)。
    """
    print(f"开始进程 {rank}/{world_size}...")
    setup(rank, world_size)

    # --- 数据集与模型定义 ---
    # 定义模型输出目录
    output_dir = "msra_ner_bert_chinese"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    # 加载数据集
    ds_raw = load_dataset('doushabao4766/msra_ner_k_V3')

    # 定义标签体系
    msra_entities = ['O', 'PER', 'ORG', 'LOC']
    tags = ['O'] + [f'{prefix}-{entity}' for entity in msra_entities[1:] for prefix in ['B', 'I']]
    id2lbl = {i: tag for i, tag in enumerate(tags)}
    lbl2id = {tag: i for i, tag in enumerate(tags)}
    
    # --- 数据预处理 ---
    def filter_no_entities(item):
        # 过滤掉 ner_tags 全为'O' (即0) 的样本，这可以加速训练
        return not all(tag == 0 for tag in item['ner_tags'])

    ds_filtered = ds_raw.filter(filter_no_entities)
    if rank == 0:
        print(f"原始数据集大小: { {split: len(ds_raw[split]) for split in ds_raw.keys()} }")
        print(f"过滤后数据集大小: { {split: len(ds_filtered[split]) for split in ds_filtered.keys()} }")

    def data_input_proc(item):
        tokenized_inputs = tokenizer(
            item['tokens'],
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=False,
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
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels_batch.append(label_ids)
        tokenized_inputs["labels"] = labels_batch
        return tokenized_inputs

    ds_processed = ds_filtered.map(data_input_proc, batched=True, remove_columns=ds_filtered['train'].column_names)
    
    # --- 训练参数 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        lr_scheduler_type='linear',
        eval_strategy="epoch",      # 每个epoch结束后进行评估
        save_strategy="epoch",      # 每个epoch结束后保存模型
        save_total_limit=2,             # 最多保留2个checkpoint
        load_best_model_at_end=True,    # 训练结束后加载最佳模型
        metric_for_best_model="f1",     # 使用f1分数作为评判最佳模型的标准
        report_to='tensorboard',
        fp16=True,                      # 开启混合精度训练
        local_rank=rank,                # **DDP关键参数**：指定当前进程的rank
        ddp_find_unused_parameters=False,
        save_safetensors=False,         # 保存为 pytorch_model.bin
    )

    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        num_labels=len(tags),
        id2label=id2lbl,
        label2id=lbl2id
    )

    # --- 评估指标 ---
    seqeval_metric = evaluate.load('seqeval')
    def compute_metrics(p):
        predictions, labels = p.predictions, p.label_ids
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [tags[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # 数据整理器
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- 初始化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_processed['train'],
        eval_dataset=ds_processed['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # --- 开始训练 ---
    print(f"进程 {rank}: 开始训练...")
    trainer.train()

    # --- 保存最终模型 ---
    # 只有主进程(rank 0)需要执行保存操作。
    # Trainer的save_model方法会自动处理好这一点。
    if rank == 0:
        print("训练完成。正在保存最终模型...")
        # trainer.save_model 会保存模型、分词器和配置文件到 output_dir
        # 因为设置了 load_best_model_at_end=True，这里保存的是效果最好的模型
        trainer.save_model(output_dir)
        print(f"模型已保存至: {output_dir}")

    # 清理分布式环境
    cleanup()


# --- 3. DDP启动入口 ---
def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"检测到 {world_size} 个GPU，将启动DDP训练。")
        mp.spawn(train_process,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    elif world_size == 1:
        print("检测到 1 个GPU，将启动单卡训练。")
        train_process(0, 1)
    else:
        print("没有检测到可用的GPU，请检查PyTorch和CUDA环境。")


if __name__ == "__main__":
    main()