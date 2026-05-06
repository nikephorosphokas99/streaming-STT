
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_model, block_size):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
