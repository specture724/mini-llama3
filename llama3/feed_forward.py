import torch
from torch import nn
import torch.nn.functional as F
from param import ModelArgs
from llama3 import device

class FeedForward(nn.module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super.__init__()
        self.dim = dim

        hidden_dim = int(2 * hidden_dim/3)
        if ffn_dim_multiplier is not None:  
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)  
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))