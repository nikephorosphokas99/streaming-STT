from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data import random_split



# dataset = load_dataset("roneneldan/TinyStories")
# train_ds = dataset["train"]
# val_ds = dataset["validation"]

# train_loader = DataLoader(train_ds, batch_size=16)
# val_loader = DataLoader(val_ds, batch_size=16)
# print("train_loader:", len(train_loader))
# print("val_loader:", len(val_loader))


class QuasLLMDataset(Dataset):
    def __init__(self, tokenizer_instance, split="train"):
        self.tokenizer = tokenizer_instance
        self.split = split
        self.dataset = load_dataset("roneneldan/TinyStories")[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoded = self.tokenizer(item["text"], return_tensors="pt")
        return encoded["input_ids"].squeeze(), encoded["attention_mask"].squeeze()
