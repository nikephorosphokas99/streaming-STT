import torch
from datasets import load_dataset
from tokenizer import BPETokenizer, TokenizerConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class QuasLLMDataset(Dataset):
    def __init__(self, splitted_ds, tokenizer_instance: BPETokenizer):
        self.tokenizer = tokenizer_instance
        self.dataset = (
            splitted_ds if isinstance(splitted_ds, list) else list(splitted_ds["text"])
        )
        self.sos_token_id = self.tokenizer.get_special_tokens()["[SOS]"]
        self.eos_token_id = self.tokenizer.get_special_tokens()["[EOS]"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.dataset[idx])
        tokens.insert(0, self.sos_token_id)
        tokens.append(self.eos_token_id)
        # print("decoded:", self.tokenizer.decode(tokens))
        return tokens
        # item = self.dataset[idx]
        # encoded = self.tokenizer(item["text"], return_tensors="pt")
        # return encoded["input_ids"].squeeze(), encoded["attention_mask"].squeeze()


def collate_fn(batch):
    batch = [torch.tensor(item) for item in batch]
    batch = pad_sequence(
        batch, batch_first=True, padding_value=BPETokenizer.get_special_tokens()["[PAD]"]
    )
    return batch

# tokenizer_config = TokenizerConfig()
# tokenizer = BPETokenizer(tokenizer_config)
# dataset = load_dataset("roneneldan/TinyStories")
# train_dataset = QuasLLMDataset(dataset["train"], tokenizer)
# val_dataset = QuasLLMDataset(dataset["validation"], tokenizer)


# def collate_fn(batch):
#     batch = [torch.tensor(item) for item in batch]
#     batch = pad_sequence(
#         batch, batch_first=True, padding_value=tokenizer.get_special_tokens()["[PAD]"]
#     )
#     return batch


# train_dataloader = DataLoader(
#     train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
# )
# val_dataloader = DataLoader(
#     val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
# )

# sample_0 = next(iter(train_dataloader))
# print("sample_0:", sample_0)
# for i in sample_0:
#     i = i.tolist()
#     print("i:", i)
#     print("decoded:", tokenizer.decode(i))
