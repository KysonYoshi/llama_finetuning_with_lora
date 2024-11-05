import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# 1. 加载预训练模型和分词器
model_name = "Qwen/Qwen2.5-7B-Instruct"  # 请根据实际模型名称调整
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

# 确保 tokenizer 能处理 PAD token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))

# 2. 创建原始模型的副本并冻结参数
original_model = copy.deepcopy(model)
original_model.eval()
for param in original_model.parameters():
    param.requires_grad = False

# 3. 配置 LoRA
lora_config = LoraConfig(
    r=4,  # 低秩矩阵的秩
    lora_alpha=16,  # LoRA alpha 参数
    lora_dropout=0.1,  # dropout 比例
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # 需要应用 LoRA 的模块
    task_type=TaskType.CAUSAL_LM
)

# 获取 PEFT 配置
peft_model = get_peft_model(model, lora_config)

# 4. 准备数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 定义预处理函数
def preprocess_function(examples):
    # 为了生成摘要，通常使用 "summarize: {article}" 作为输入
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # 编码目标摘要
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=512, truncation=True, padding="max_length")
    
    # 将标签中填充的 token 设置为 -100，以忽略损失计算
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] 
        for labels_example in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 应用预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])

# 5. 定义数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True, pad_to_multiple_of=64)

# 6. 定义自定义 Trainer 以包含前五层输出一致性的损失
# 6. 定義自訂 Trainer 以包含前五層輸出一致性的損失
class CustomTrainer(Trainer):
    def __init__(self, original_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model = original_model.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 微调后的模型进行前向传播
        outputs = model(**inputs, output_hidden_states=True)
        loss = outputs.loss

        # # 使用原始模型计算前五层的隐藏状态
        # with torch.no_grad():
        #     original_outputs = self.original_model(**inputs, output_hidden_states=True)
        
        # 获取微调模型的前五层隐藏状态
        fine_tuned_hidden_states = outputs.hidden_states[5:6]
        # 获取原始模型的前五层隐藏状态
        # original_hidden_states = original_outputs.hidden_states[:5]
        
        # 计算前五层的输出一致性损失（使用MSE损失）
        consistency_loss = 0
        for ft_hidden, orig_hidden in zip(fine_tuned_hidden_states, outputs.hidden_states[-1:]):
            consistency_loss += torch.nn.functional.mse_loss(ft_hidden, orig_hidden)
        
        # 总损失 = 原始训练损失 + 一致性损失
        total_loss = loss + 1e-4 * consistency_loss

        return (total_loss, outputs) if return_outputs else total_loss


# 7. 设置训练参数
training_args = TrainingArguments(
    output_dir="./qwen_lora_finetuned_cnndm",
    per_device_train_batch_size=1,  # 根据显存调整
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # 增加梯度累积以模拟更大的批次
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    fp16=True,  # 如果使用 GPU 支持半精度训练
    save_total_limit=2,  # 最多保留2个检查点
    #predict_with_generate=True,  # 生成摘要进行评估
    #ddp_find_unused_parameters=False,
)

# 8. 定义 Trainer
trainer = CustomTrainer(
    original_model=original_model,
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 9. 开始训练
trainer.train()

# 10. 保存微调后的模型
peft_model.save_pretrained("./qwen_lora_finetuned_cnndm")
tokenizer.save_pretrained("./qwen_lora_finetuned_cnndm")
