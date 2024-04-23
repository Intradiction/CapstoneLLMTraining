from transformers import pipeline, AutoTokenizer, BertForSequenceClassification
from peft import PeftModel

base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model_id = "Intradiction/BERT-SA-LORA"
model = PeftModel.from_pretrained(model=base_model, model_id=peft_model_id)
merged_model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

lorapipe = pipeline("sentiment-analysis", model=merged_model, tokenizer=tokenizer)
print(lorapipe('This movie is awesome'))