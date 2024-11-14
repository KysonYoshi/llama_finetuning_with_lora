import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-3B-Instruct" 
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))

original_model = copy.deepcopy(model)
original_model.eval()
for param in original_model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=4,  
    lora_alpha=16,  
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  
    task_type=TaskType.CAUSAL_LM
)


peft_model = get_peft_model(model, lora_config)

dataset = load_dataset("cnn_dailymail", "3.0.0")

def preprocess_function(examples):

    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=512, truncation=True, padding="max_length")
    
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] 
        for labels_example in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True, pad_to_multiple_of=64)

class CustomTrainer(Trainer):
    def __init__(self, original_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model = original_model.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        outputs = model(**inputs, output_hidden_states=True)
        loss = outputs.loss

        # with torch.no_grad():
        #     original_outputs = self.original_model(**inputs, output_hidden_states=True)
        
        fine_tuned_hidden_states = outputs.hidden_states[5:6]
        
        consistency_loss = 0
        for ft_hidden, orig_hidden in zip(fine_tuned_hidden_states, outputs.hidden_states[-1:]):
            consistency_loss += torch.nn.functional.mse_loss(ft_hidden, orig_hidden)
        
        
        total_loss = loss + 1e-4 * consistency_loss

        return (total_loss, outputs) if return_outputs else total_loss



training_args = TrainingArguments(
    output_dir="./qwen_lora_finetuned_cnndm",
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    fp16=True,  
    save_total_limit=2,  
    #predict_with_generate=True,  
    #ddp_find_unused_parameters=False,
)

trainer = CustomTrainer(
    original_model=original_model,
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

peft_model.save_pretrained("./qwen_lora_finetuned_cnndm")
tokenizer.save_pretrained("./qwen_lora_finetuned_cnndm")
