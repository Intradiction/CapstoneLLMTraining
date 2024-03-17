from huggingface_hub import login
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

import numpy as np
import evaluate

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

login("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")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#imdb = load_dataset("imdb")
#tokenized_imdb = imdb.map(preprocess_function, batched=True)

tokenized_datasets = load_from_disk("./imdb-tokenized-datasets")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(3000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(3000))
print(small_train_dataset.features)

accuracy = evaluate.load("accuracy")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=1, target_modules=["query", "key", "value"], lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="BERT-SA-LORA",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()