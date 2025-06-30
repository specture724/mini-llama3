import torch
from torch import nn
import torch.nn.functional as F
from param import ModelArgs
from llama3 import device
from llama3.RoPE import apply_rotary_emb, precompute_freqs_cis
import math

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, inference):
        super.__init__()
        self.args = args
        # Embedding dim
        self.dim = args.dim
        # Q num
        self.n_heads = args.n_heads
        # KV num. GQA: if n_kv_heads is None, classic attention is used, else GQA.
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # The output dim for each head. After concating, the total output dim equals the embedding dim
        self.head_dim = args.dim // args.n_heads
        # The number of Qs for each KV
        self.n_rep = args.n_heads // args.n_kv_heads

        self.inference = inference
        # The weights
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=device)

        # KV cache
        self.k_cache = torch.zeros((args.max_batch_size, 
                                    args.max_seq_len, 
                                    self.n_kv_heads, 
                                    args.dim), 
                                    device=device)
        self.v_cache = torch.zeros((args.max_batch_size, 
                                    args.max_seq_len, 
                                    self.n_kv_heads, 
                                    args.dim), 
                                    device=device)
        if inference:
            self.freq_cis = precompute_freqs_cis(self.head_dim, self.args.max_seq_len*2)
        else:
            self.freq_cis = precompute_freqs_cis(self.head_dim, self.args.max_seq_len)
        
    def forward(self, x:torch.Tensor, start_pos):
        '''
        x -> [batch_size, seq_len, dim]
        wq -> [dim, n_heads * head_dim]
        wk, wv -> [dim, n_kv_heads * head_dim] 
        xq -> [batch_size, seq_len, n_heads * head_dim] -> [batch_size, seq_len, n_heads, head_dim]
        xk, xv -> [batch_size, seq_len, n_kv_heads * head_dim] -> [batch_size, seq_len, n_kv_heads, head_dim]
        '''
        batch_size, seq_len = x.shape
        mask = None

        queries = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # inference with kv cache
        if self.inference:    
            freq_cis = self.freq_cis[start_pos: start_pos + seq_len]
            queries, xk = apply_rotary_emb(queries, xk, freq_cis)
            self.k_cache = self.k_cache.to(queries)
            self.v_cache = self.k_cache.to(queries)
            
            # Apply kv cache
            # store the current xk & xv to the cache
            self.k_cache[:batch_size, start_pos: start_pos + seq_len] = xk
            self.v_cache[:batch_size, start_pos: start_pos + seq_len] = xv
            
            # fetch the former cached kvs from kv cache
            keys = self.k_cache[:batch_size, :start_pos + seq_len]
            values = self.v_cache[:batch_size, :start_pos + seq_len]

            # keys/values have different shape with queries, apply repeat_kv to repeat keys/values to the shape of queries
            keys = self.repeat_kv(keys, self.n_rep)
            values = self.repeat_kv(values, self.n_rep)
        
        # training with mask and without kv cache
        else:
            queries, xk = apply_rotary_emb(queries, xk, self.freq_cis)
            keys = self.repeat_kv(xk, self.n_rep)
            values = self.repeat_kv(xv, self.n_rep)
            mask = torch.full((seq_len, seq_len),float("-inf"),device=self.args.device)  
            mask = torch.triu(mask, diagonal=1).to(self.args.device)  

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(2, 3)).to(device) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values).to(device)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)

    def repeat_kv(x:torch.Tensor, n_rep:int):
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return(
                x[:, :, :, None, :]
                .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
            )

