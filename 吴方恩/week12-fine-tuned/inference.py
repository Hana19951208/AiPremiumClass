# inference.py

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

def extract_entities_from_text(text, model, tokenizer):
    """
    从给定文本中抽取命名实体。
    此函数直接从你的原始脚本中提取和优化。
    """
    # 将模型置于评估模式
    model.eval()
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. 分词并准备模型输入
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. 模型预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. 获取预测结果
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_label_ids = predictions[0].cpu().tolist()
    input_ids = inputs["input_ids"][0].cpu().tolist()

    # 4. 将标签ID转换为标签字符串，并与原始词元对齐
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 从模型配置中获取 id 到 label 的映射
    id2label_map = model.config.id2label

    # 过滤掉特殊token [CLS], [SEP], [PAD]
    meaningful_tokens = []
    meaningful_tags = []
    special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

    for token_id, label_id in zip(input_ids, predicted_label_ids):
        if token_id not in special_token_ids:
            meaningful_tokens.append(tokenizer.decode([token_id]))
            meaningful_tags.append(id2label_map[label_id])
        if token_id == tokenizer.sep_token_id:
            break
            
    # 5. 聚合实体 (BIO标注解码)
    entities = []
    current_entity_content = []
    current_entity_label = None

    for token_char, tag_str in zip(meaningful_tokens, meaningful_tags):
        if tag_str.startswith("B-"):
            if current_entity_content:
                entities.append({
                    "entity_type": current_entity_label,
                    "content": "".join(current_entity_content)
                })
            current_entity_label = tag_str[2:]
            current_entity_content = [token_char]
        elif tag_str.startswith("I-"):
            if current_entity_label and tag_str[2:] == current_entity_label:
                current_entity_content.append(token_char)
            else: # I标签与B不匹配或无B，作容错处理
                if current_entity_content:
                    entities.append({
                        "entity_type": current_entity_label,
                        "content": "".join(current_entity_content)
                    })
                current_entity_label = tag_str[2:]
                current_entity_content = [token_char]
        else: # O标签
            if current_entity_content:
                entities.append({
                    "entity_type": current_entity_label,
                    "content": "".join(current_entity_content)
                })
            current_entity_content = []
            current_entity_label = None
            
    if current_entity_content:
        entities.append({
            "entity_type": current_entity_label,
            "content": "".join(current_entity_content)
        })
        
    return entities

def main():
    # --- 模型加载 ---
    # 指定训练脚本保存的模型目录
    model_path = "msra_ner_bert_chinese"
    print(f"正在从 '{model_path}' 加载模型和分词器...")

    try:
        # AutoClass.from_pretrained 会自动加载目录中的配置文件、权重和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        print("模型加载成功！")
    except OSError:
        print(f"错误：在 '{model_path}' 找不到模型文件。请先运行 train_ddp.py 完成训练。")
        return

    # --- 推理演示 ---
    print("\n--- 开始进行实体抽取推理 ---")
    test_sentences = [
        "中华人民共和国国务院总理李强访问德国。",
        "双方确定了今后发展中美关系的指导方针。",
        "我最喜欢的城市是北京和上海。",
        "阿里巴巴集团在杭州宣布了新的战略规划。",
    ]

    for sentence in test_sentences:
        extracted_entities = extract_entities_from_text(sentence, model, tokenizer)
        print(f"\n输入: \"{sentence}\"")
        print(f"输出: {extracted_entities}")

if __name__ == "__main__":
    main()