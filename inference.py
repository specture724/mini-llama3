from llama3.input_block import input_block
from llama3.llama3 import Transformer
import torch
from llama3 import device
from llama3.param import ModelArgs, params
import numpy as np
import time
import pandas as pd

def generate(model, prompts: str, params: ModelArgs, max_gen_len: int=500, temperature: float = 0.6, top_p: float = 0.9):  

    # prompt_tokens: 用户输入文本或提示列表  
    # max_gen_len: 生成文本序列的最大长度  
    # temperature: 用于控制采样随机性的温度值。默认为0.6  
    # top_p: 从logits采样prob输出的top-p概率阈值。默认为0.9  
    bsz = 1  # 对于推理，通常用户只输入一个提示，我们将其作为1个批次  
    prompt_tokens = input_block.token_bos.tolist() + input_block.encode(prompts)  
    assert len(prompt_tokens) <= params.max_seq_len, "提示标记长度应小于max_seq_len"  
    total_len = min(len(prompt_tokens)+max_gen_len, params.max_seq_len)    

    # 这个tokens矩阵用于存储输入提示和模型生成的所有输出  
    # 稍后我们将使用分词器的decode函数来解码这个token，以文本格式查看结果  
    tokens = torch.full((bsz,total_len), fill_value=input_block.token_pad.item(), dtype=torch.long, device=device)  

    # 将提示tokens填入token矩阵  
    tokens[:,:len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)  

    # 创建一个prompt_mask_token，用于稍后识别token是提示token还是填充token  
    # 如果是提示token则为True，如果是填充token则为False  
    input_text_mask = tokens != input_block.token_pad.item()  

    # 现在我们可以从第一个位置开始，一次使用一个token从prompt_tokens列表开始推理  
    prev_pos = 0  
    for cur_pos in range(1, total_len):  
        with torch.no_grad():  
            logits, _ = model(x=tokens[:,prev_pos:cur_pos], start_pos=prev_pos)  
        if temperature > 0:        
            probs = torch.softmax(logits[:, -1]/temperature, dim=-1)  
            next_token = sample_top_p(probs, top_p)          
        else:  
            next_token = torch.argmax(logits[:, -1], dim=-1)          

        next_token = next_token.reshape(-1)  

        # 只有在是填充token时才替换token  
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)  
        tokens[:, cur_pos] = next_token  

        prev_pos = cur_pos  
        if tokens[:,cur_pos]==input_block.token_pad.item() and next_token == input_block.token_eos.item():  
            break  

    output_tokens, output_texts = [], []      

    for i, toks in enumerate(tokens.tolist()):  
        if input_block.token_eos.item() in toks:  
            eos_idx = toks.index(input_block.token_eos.item())  
            toks = toks[:eos_idx]  
        output_tokens.append(toks)  
        output_texts.append(input_block.decode(toks))  

    return output_tokens, output_texts  

def sample_top_p(probs, top_p):
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)  
    probs_sum = torch.cumsum(probs_sort, dim=-1)  
    mask = probs_sum - probs_sort > top_p  
    probs_sort[mask] = 0.0  
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  
    next_token = torch.multinomial(probs_sort, num_samples=1)  
    next_token = torch.gather(prob_idx, -1, next_token)      
    # 返回从词汇表中采样的标记索引  
    return next_token

prompts = "Consider you what services he has done"
model = Transformer(params, inference=True).to(device)
model.load_state_dict(torch.load("models/llama3_model.pth", map_location=device))  
model.eval()
while True:
    try:
        prompts = input("Enter your prompt (or 'exit' to quit): ")
        if prompts.lower() == 'exit':
            break
    except EOFError:
        break
    if not prompts.strip():
        continue
    start_time = time.time()
    output_tokens, output_texts = generate(model, prompts, params)
    end_time = time.time()
    output_texts = output_texts[0].replace("<|begin_of_text|>", "")  
    print(f"Generated text: {output_texts}")
