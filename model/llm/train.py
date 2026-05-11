from datasets import load_dataset
from torch.utils.data import DataLoader
from model import Config, QuasLLM
from tokenizer import BPETokenizer, TokenizerConfig
import os
import torch


config = Config()
tokenizer_config = TokenizerConfig()


dataset = load_dataset("roneneldan/TinyStories")
train_ds = dataset["train"]
test_ds = dataset["validation"]

tokenizer = BPETokenizer(tokenizer_config)
if not os.path.exists(tokenizer_config.module_path_dir):
    print("training tokenizer...")
    tokenizer.train()
else:
    print("loaded tokenizer...")


train_loader = DataLoader(train_ds, batch_size=config.batch_size)
test_loader = DataLoader(test_ds, batch_size=config.batch_size)

# model & optimizers

model = QuasLLM(config).to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
