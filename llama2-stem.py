import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Trainer
)
import pandas as pd
from peft import LoraConfig, PeftModel

model_name = "meta-llama/Llama-2-7b-hf"

dataset = pd.read_csv('10000entities.csv')

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_text(text, entities, index):
  # Tokenize the text
    tokenized_text = tokenizer.tokenize(text)

    # Initialize labels for each token
    token_labels = ['O'] * len(tokenized_text)  # 'O' indicates that the token is not part of any entity

    # Assign labels for each entity in the text
    for entity, entity_type in entities:
        entity_tokens = tokenizer.tokenize(entity)
        start_idx = 0
        while start_idx < len(tokenized_text):
            try:
                start_idx = tokenized_text.index(entity_tokens[0], start_idx)
                end_idx = start_idx + len(entity_tokens)
                if tokenized_text[start_idx:end_idx] == entity_tokens:
                    token_labels[start_idx:end_idx] = [f'B-{entity_type}'] + [f'I-{entity_type}'] * (len(entity_tokens) - 1)
                    start_idx = end_idx
                else:
                    start_idx += 1
            except ValueError:
                break

    df['ner_tags'][index] = entity_tokens

df['tokenized_text'] = df.apply(lambda row: tokenize_text(row['trimmed_text'], row['entities'], row), axis=1)

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False


output_dir = "./results"

num_train_epochs = 1
fp16 = False
bf16 = False

per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03

group_by_length = True
save_steps = 25
logging_steps = 25
max_seq_length = None
packing = False

device_map = {"": 0}

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

eval_dataset = load_dataset('adsabs/WIESP2022-NER')

# Set supervised fine-tuning parameters
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    eval_dataset=eval_dataset
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
