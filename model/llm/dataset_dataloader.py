from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data import random_split



dataset = load_dataset("roneneldan/TinyStories")
train_ds = dataset["train"]
val_ds = dataset["validation"]

train_loader = DataLoader(train_ds, batch_size=16)
val_loader = DataLoader(val_ds, batch_size=16)
print("train_loader:", len(train_loader))
print("val_loader:", len(val_loader))
