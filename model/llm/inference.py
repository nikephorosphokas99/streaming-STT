import torch
from safetensors.torch import load_file

from model import Config, QuasLLM
from tokenizer import BPETokenizer, TokenizerConfig

checkpoint_path = "model.safetensors"

device = "cuda"

config = Config()
model = QuasLLM(config).to(device)

state_dict = load_file(checkpoint_path)
model.load_state_dict(state_dict)
model.eval()

from torch.nn import functional as F

tokenizer_config = TokenizerConfig()
tokenizer = BPETokenizer(tokenizer_config)


@torch.no_grad()
def generate(text: str, max_new_tokens: int = 500, temperature: float = 1.0):

    ids = tokenizer.encode(text)
    ids = torch.tensor(ids).to(device)
    ids = ids.unsqueeze(0)

    for _ in range(max_new_tokens):
        ids = ids if ids.size(1) <= config.block_size else ids[:, -config.block_size :]

        logits = model(ids)  # (1, T, vocab_size)
        # print("logits_shape:", logits.shape)
        last_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1)
        ids_next = torch.multinomial(probs, num_samples=1)
        ids_next = ids_next.unsqueeze(0)
        # print("ids_next:", ids_next)
        ids = torch.cat([ids, ids_next], dim=1)
        if ids_next.item() == tokenizer.get_special_tokens()["[EOS]"]:
            break

    return ids


text = "Once upon a"
ids = tokenizer.encode(text)
result = generate(text)  # int token id
# ids.append(next_tok)  # mutates ids in-place; don't capture the None return
print("generate result:", result)
decoded_result = tokenizer.decode(result.squeeze(0).tolist(), format_result=False)
# print()
print("decoded_result:", decoded_result)
# print("next_tok decoded:", tokenizer.decode([next_tok]))
# print("text:", text)
# print("=" * 10)
# print("whole sentence:", tokenizer.decode(ids))

# with torch.no_grad():
