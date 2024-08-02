import torch
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = load_dataset("emotion", trust_remote_code=True)
    metric = evaluate.load("accuracy")
    
    num_labels = 6
    id2label = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    label2id = {v: k for k, v in id2label.items()}
    
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1)
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id).to(device)
    model = get_peft_model(model, lora_config)

    dataset_tokenized = dataset.map(lambda batch: tokenize_function(tokenizer, batch), batched=True)
    train_dataset = dataset_tokenized["train"].shuffle(seed=42)
    validation_dataset = dataset_tokenized["validation"].shuffle(seed=42)
    
    batch_size = 64
    training_args = TrainingArguments(
        output_dir=f"{model_name}-emotion", 
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        logging_dir='./logs',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=lambda p: compute_metrics(metric, p),
        tokenizer=tokenizer
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
