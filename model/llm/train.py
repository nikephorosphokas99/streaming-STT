import os

import torch
from dataset_dataloader import QuasLLMDataset, collate_fn
from datasets import load_dataset
from safetensors.torch import save_file
from tokenizer import BPETokenizer, TokenizerConfig
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

# from torch.nn.utils import clip_grad_norm_
from model import Config, QuasLLM

config = Config()
tokenizer_config = TokenizerConfig()
device = "cuda"

tokenizer = BPETokenizer(tokenizer_config)
model = QuasLLM(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

criterion = torch.nn.CrossEntropyLoss(ignore_index=3)
# dataset & dataloaders
dataset = load_dataset("roneneldan/TinyStories")
train_dataset = QuasLLMDataset(dataset["train"], tokenizer)
val_dataset = QuasLLMDataset(dataset["validation"], tokenizer)
train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
)

for epoch in range(config.epochs):
    pbar = tqdm(train_dataloader)
    for i, batch in enumerate(pbar):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        y_preds = logits.transpose(-2, -1)
        loss = criterion(y_preds, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
print("saving safetensors....")
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")
# batch = next(iter(train_dataloader))
# print(batch)
