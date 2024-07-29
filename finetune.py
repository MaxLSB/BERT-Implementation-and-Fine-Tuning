# Finetuning BERT for sentiment classification and emotion classification
# Deploy on Streamlit

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model


def tokenize_function(tokenizer, batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(dataset, tokenizer, metric, model, n_epochs):
    
    dataset_tokenized = dataset.map(tokenize_function(tokenizer), batched=True)
    train_dataset = dataset_tokenized["train"].shuffle(seed=42)
    test_dataset = dataset_tokenized["test"].shuffle(seed=42)
    
    # small_train_dataset = dataset_tokenized["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset = dataset_tokenized["test"].shuffle(seed=42).select(range(1000))
    
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1)
    
    # 6 labels : anger, fear, joy, love, sadness, surprise
    model = BertForSequenceClassification.from_pretrained(model, num_labels=6)
    model = get_peft_model(model, lora_config)

    # training_args = TrainingArguments(
    #     output_dir="test_trainer", 
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     disable_tqdm=False
    #     push_to_hub=True,
    #     weight_decay=0.01,
    #     num_train_epochs=n_epochs
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics(metric),
        tokenizer=tokenizer
    )
    
    trainer.train()
    
def __init__():
    
    model = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model) # AutoTokenizer could also be used here
    dataset = load_dataset("emotion", trust_remote_code=True)
    metric = evaluate.load("accuracy")
    
    n_epochs = 20
    
    main(dataset, tokenizer, metric, n_epochs)
    