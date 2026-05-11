from datasets import load_dataset
from torch.utils.data import DataLoader
from model import Config


config = Config()
dataset = load_dataset("roneneldan/TinyStories")
train_ds = dataset["train"]
test_ds = dataset["test"]

train_loader = DataLoader(train_ds, batch_size=config.batch_size)
test_loader = DataLoader(test_ds, batch_size=config.batch_size)
