import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())