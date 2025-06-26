import torch
from llama3.param import params
from llama3 import device
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=params.norm_eps):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.eps = eps

    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(device)
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight