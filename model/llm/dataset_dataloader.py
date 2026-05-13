import torch
import tqdm
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
        x = tokens[:-1]
        y = tokens[1:]
        # print("decoded:", self.tokenizer.decode(tokens))
        return torch.tensor(x), torch.tensor(y)
        # item = self.dataset[idx]
        # encoded = self.tokenizer(item["text"], return_tensors="pt")
        # return encoded["input_ids"].squeeze(), encoded["attention_mask"].squeeze()


def collate_fn(batch):
    # padded_batch = []

    x = [item[0] for item in batch]
    y = [item[1] for item in batch]

    # print(batch.shape)
    x = pad_sequence(
        x, batch_first=True, padding_value=BPETokenizer.get_special_tokens()["[PAD]"]
    )
    y = pad_sequence(
        y, batch_first=True, padding_value=BPETokenizer.get_special_tokens()["[PAD]"]
    )
    return x, y
    # for (x, y) in batch:
    #     x = torch.tensor(x)
    #     y = torch.tensor(y)
        # print("X:", x.shape)
        # print("Y:", y.shape)
        # x = pad_sequence(
        #     x, batch_first=True, padding_value=BPETokenizer.get_special_tokens()["[PAD]"]
        # )
        # y = pad_sequence(
        #     y, batch_first=True, padding_value=BPETokenizer.get_special_tokens()["[PAD]"]
        # )
        # padded_batch.append((x, y))
    return padded_batch

# def collate_fn(batch):
#     batch = [torch.tensor(item) for item in batch]
#     batch = pad_sequence(
#         batch, batch_first=True, padding_value=tokenizer.get_special_tokens()["[PAD]"]
#     )
#     return batch


# tokenizer_config = TokenizerConfig()
# tokenizer = BPETokenizer(tokenizer_config)
# dataset = load_dataset("roneneldan/TinyStories")
# train_dataloader = DataLoader(
#     train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
# )
# val_dataloader = DataLoader(
#     val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
# )


# train_dataset = QuasLLMDataset(dataset["train"], tokenizer)
# val_dataset = QuasLLMDataset(dataset["validation"], tokenizer)
# train_dataloader = DataLoader(
#     train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
# )
# val_dataloader = DataLoader(
#     val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
# )

# sample_0 = next(iter(train_dataloader))
# print("sample_0 len:", len(sample_0))
# print("sample_0:", sample_0[0].shape)
# for x, y in sample_0:
#     x = x.tolist()
#     y = y.tolist()
#     print("x:", x)
#     print("y:", y)
#     print("decoded x:", tokenizer.decode(x))
#     print("decoded y:", tokenizer.decode(y))
