import torch
from torch import nn
import torch.nn.functional as F
from param import ModelArgs
from llama3 import device
from llama3.Attention import Attention
from llama3.feed_forward import FeedForward

class TransformerBlock(nn.module):
    def __init__(self, args:ModelArgs, inference):
        super.__init__()
        self.args = args
        self.inference = inference
        self.attention_RMSNorm = nn.RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.attention = Attention(args, inference)
        self.ff_RMSNorm = nn.RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.feedforward = FeedForward(args.dim, 4*args.dim, args.multiple_of, args.ffn_dim_multiplier)

    def forward(self, x, start_pos):
        h = x + self.attention(self.attention_RMSNorm(x), start_pos, self.inference)
        return h + self.feedforward(self.ff_RMSNorm(h))