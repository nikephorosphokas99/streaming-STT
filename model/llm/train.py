import os

import torch
from dataset_dataloader import QuasLLMDataset, collate_fn
from datasets import load_dataset
from tokenizer import BPETokenizer, TokenizerConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils import clip_grad_norm_

from model import Config, QuasLLM

config = Config()
tokenizer_config = TokenizerConfig()
device = "cuda"

tokenizer = BPETokenizer(tokenizer_config)
model = QuasLLM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

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

# training loop
import tqdm
for epoch in range(config.epochs):
    pbar = tqdm.tqdm(train_dataloader)
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()
        y_true
        y_pred= model(batch)
        # print("batch:", batch.shape)
        # break
    break
