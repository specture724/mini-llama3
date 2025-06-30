from llama3.input_block import input_block
from llama3.llama3 import Transformer
import torch
from llama3 import device
from llama3.param import ModelArgs, params
import numpy as np
import time
import pandas as pd

with open("data/tiny_shakespeare.txt", "r") as f:
    data = f.read()

dataset = torch.tensor(input_block.encode(data), dtype=torch.int).to(device)

def get_dataset_batch(data, split, args:ModelArgs):
    seq_len = args.max_seq_len  
    batch_size = args.max_batch_size

    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)): int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    batch_data = train
    if split == "val":
        batch_data = val
    elif split == "test":
        batch_data = test
    ix = torch.randint(0, len(batch_data) - seq_len - 3, (batch_size,)).to(device)  
    x = torch.stack([torch.cat([input_block.token_bos, batch_data[i:i+seq_len-1]]) for i in ix]).long().to(device)  
    y = torch.stack([torch.cat([batch_data[i+1:i+seq_len], input_block.token_eos]) for i in ix]).long().to(device)  
     
    return x, y

@torch.no_grad
def evaluate_loss(model, args:ModelArgs):
    out = {}  
    model.eval()  
    
    for split in ["train", "val"]:  
        losses = []  
        for _ in range(10):        
            xb, yb = get_dataset_batch(dataset, split, args)  
            _, loss = model(x=xb, targets=yb)  
            losses.append(loss.item())  
        out[split] = np.mean(losses)  
    
    model.train()  
    return out

def train(model, optimizer, args:ModelArgs):
    epochs = args.epochs  
    log_interval = args.log_interval  
    losses = []
    start_time = time.time()  

    for epoch in range(epochs):
        optimizer.zero_grad()

        xs, ys = get_dataset_batch(dataset, 'train', args)
        xs = xs.to(device)
        ys = ys.to(device)
        logits, loss = model(x=xs, targets=ys)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:  
            batch_time = time.time() - start_time  
            x = evaluate_loss(model, args)  
            losses.append(x)              
            print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f}")  
            start_time = time.time()  
 
    print("loss: ", losses[-1]['val'])  
    plot = pd.DataFrame(losses).plot()
    plot.figure.savefig('training_loss.png')
    return plot

model = Transformer(params, inference=False).to(device)  
optimizer = torch.optim.Adam(model.parameters())  

train(model, optimizer, params)
# Save the model
torch.save(model.state_dict(), "models/llama3_model.pth")
