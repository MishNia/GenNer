import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load varied domain, annotated dataset
df = pd.read_csv('10000entities.csv')

# Define tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_text(text, entities, index):
  # Tokenize the text
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
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

    # # Print tokenized text with corresponding labels
    # for token, label in zip(tokenized_text, token_labels):
    #     print(f"{token}: {label}")

df['tokenized_text'] = df.apply(lambda row: tokenize_text(row['trimmed_text'], row['entities'], row), axis=1)

config = GPT2Config.from_pretrained("gpt2", num_labels=len(df["ner_tags"]))
model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=config)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
)

# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {
        "f1": f1
    }

#evaluation dataset - STEM dataset
eval_dataset = load_dataset('adsabs/WIESP2022-NER')

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune model
trainer.train()

# Evaluate model
results = trainer.evaluate()

print(results)
