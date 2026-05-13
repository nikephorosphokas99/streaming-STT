import os

import torch
from dataset_dataloader import QuasLLMDataset, collate_fn
from datasets import load_dataset
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
        # print("decoded x:", tokenizer.decode(batch[0][0].tolist()))
        # print("decoded y:", tokenizer.decode(batch[1][0].tolist()))
        # print(f"batch_{i}:")
        # print(len(batch))
        # print("x:", batch[0].shape)
        # print("y:", batch[1].shape)
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        # print(f"processing batch {i}")
        # print(f"x: {x.shape}, y: {y.shape}")
        optimizer.zero_grad()
        logits = model(x)
        y_preds = logits.transpose(-2, -1)
        # print("logits:", logits.shape)
        # print("y_preds:", y_preds.shape)
        loss = criterion(y_preds, y)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # loss =
        # print("y_pred:", y_pred.shape)

        # break
    # break


# batch = next(iter(train_dataloader))
# print(batch)
