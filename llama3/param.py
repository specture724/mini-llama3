import torch
from dataclasses import dataclass
from typing import Optional
from llama3.input_block import input_block

@dataclass  
class ModelArgs:  
    dim: int = 512                              # 嵌入维度  
    n_layers: int = 8                           # 模型解码器块的数量  
    n_heads: int = 8                            # q头数  
    n_kv_heads: int = 4                         # kv头数  
    vocab_size: int = len(input_block.vocab)    # 词汇表长度  
    multiple_of: int = 256                      # 用于计算前馈网络维度  
    ffn_dim_multiplier: Optional[float] = None  # 用于计算前馈网络维度  
    norm_eps: float = 1e-5                      # RMSNorm计算的默认Epsilon值  
    rope_theta: float = 10000.0                 # RePE计算的默认theta值  

    max_batch_size: int = 10                    # 最大批量大小  
    max_seq_len: int = 256                      # 最大序列长度  

    epochs: int = 2500                          # 总训练迭代次数  
    log_interval: int = 10                      # 打印日志和损失值的间隔数    

params = ModelArgs()