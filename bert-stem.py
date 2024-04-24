from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.roberta.modeling_roberta import (
    ROBERTA_INPUTS_DOCSTRING,
    ROBERTA_START_DOCSTRING,
    RobertaEmbeddings,
    _TOKENIZER_FOR_DOC
)
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch import nn
from transformers import DataCollatorWithPadding

df = pd.read_csv('10000entities.csv')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

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

class SpanCategorization(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
    
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

training_args = TrainingArguments(
    output_dir="./models/fine_tune_bert_output_span_cat",
    evaluation_strategy="epoch",
    learning_rate=2.5e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_steps = 100,
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    log_level='critical',
    seed=12345
)

def model_init():
    return SpanCategorization.from_pretrained("google-bert/bert-base-cased", id2label=id2label, label2id=label2id)

eval_dataset = load_dataset('adsabs/WIESP2022-NER')

eval_dataset['tokenized_text'] = eval_dataset.apply(lambda row: tokenize_text(row['text'], row['ner_tags'], row), axis=1)

data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {
        "f1": f1
    }

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=df,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()