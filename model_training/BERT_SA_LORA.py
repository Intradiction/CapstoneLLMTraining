from huggingface_hub import notebook_login
from datasets import load_dataset

imdb = load_dataset("imdb")
notebook_login()