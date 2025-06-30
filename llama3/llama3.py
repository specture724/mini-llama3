import torch
from torch import nn
import torch.nn.functional as F
from param import ModelArgs
from llama3 import device
from llama3.transformer import TransformerBlock
from llama3.input_block import InputBlock
from llama3.RMSNorm import RMSNorm

class Transformer(nn.module):
    def __init__(self, args:ModelArgs, inference):
        super.__init__()
        self.args = args
        self.inference = inference
        self.token_embedding = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args, inference))
        self.norm = RMSNorm(args.dim, eps = args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x, start_pos=0, targets=None):
        if self.inference:
            assert targets is None
        else:
            assert targets is not None

        h = self.token_embedding(x)

        for layer in self.layers:
            h = layer(h, start_pos)
        
        h = self.norm(h)

        logits = self.output(h).float()
        loss = None

        if self.inference:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, self.args.vocab_size), targets.view(-1))

        return logits, loss
