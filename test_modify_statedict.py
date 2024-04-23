from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, pipeline
from peft import PeftModel
import torch

base_model2 = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-xsmall", ignore_mismatched_sizes=True)
current_model_dict = base_model2.state_dict()

print(current_model_dict)

loaded_state_dict = torch.load(path, map_location=torch.device('cpu'))
new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}


peft_model_id2 = "rajevan123/STS-Lora-Fine-Tuning-Capstone-Deberta-old-model-pipe-test_augmentation"
model2 = PeftModel.from_pretrained(model=base_model2, model_id=peft_model_id2)

sa_merged_model2 = model2.merge_and_unload()
bbu_tokenizer2 = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")


DebertawithLORA_pipe = pipeline("text-classification",model=sa_merged_model2, tokenizer=bbu_tokenizer2)
DebertawithLORA_pipe({'text': 'the ball is colored', 'text_pair': 'the ball is green'})