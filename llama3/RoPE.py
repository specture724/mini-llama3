import torch
from typing import Tuple
from llama3 import device
# 生成旋转矩阵
def precompute_freqs_cis(dim, seq_len, theta = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度 theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float().to(device)    # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):  
    ndim = x.ndim  
    assert 0<=1<ndim  
    assert freqs_cis.shape == (x.shape[1],x.shape[-1]), "freqs_cis的最后两个维度必须与x匹配"  
    shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]  
    return freqs_cis.view(*shape)  

# 旋转位置编码计算
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:  
   xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)    #xq_:[bsz, seq_len, n_heads, head_dim/2]  
   xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)    #xk_:[bsz, seq_len, n_heads, head_dim/2]  
 
   freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  
 
   xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device) #xq_out:[bsz, seq_len, n_heads, head_dim]  
   xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device) #xk_out:[bsz, seq_len, n_heads, head_dim]  
   return xq_out.type_as(xq), xk_out.type_as(xk)  