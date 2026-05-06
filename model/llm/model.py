import torch
import torch.nn as nn


def apply_rope(x, freqs):
    pass


class DecoderBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, freqs):
        pass

class QuasLLM(nn.Module):
    def __init__(self):
        super().__init__()

        # self.config = config
        self.vocab_size = 1000
        self.n_layers = 6
        self.d_model = 512
        self.n_layers = 4
        self.norm_eps = 1e-5


        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout()

        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(6)])
        self.norm = RMSNorm(self.d_model, eps=self.norm_eps)
        self.output = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x, freqs)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 10000
block_size = 1024

dataset = torch.randint(0, vocab_size, (8, block_size))

model = QuasTransformer().to(device)
result = model(dataset)


print(result)
