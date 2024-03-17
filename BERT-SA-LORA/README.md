---
license: apache-2.0
library_name: peft
tags:
- generated_from_trainer
metrics:
- accuracy
base_model: bert-base-uncased
model-index:
- name: BERT-SA-LORA
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# BERT-SA-LORA

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3017
- Accuracy: 0.8943

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 8
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 375  | 0.3246          | 0.8687   |
| 0.5095        | 2.0   | 750  | 0.3154          | 0.8803   |
| 0.3046        | 3.0   | 1125 | 0.2957          | 0.888    |
| 0.2754        | 4.0   | 1500 | 0.2971          | 0.8937   |
| 0.2754        | 5.0   | 1875 | 0.3017          | 0.8943   |


### Framework versions

- PEFT 0.9.0
- Transformers 4.38.2
- Pytorch 2.2.1+cu118
- Datasets 2.18.0
- Tokenizers 0.15.2