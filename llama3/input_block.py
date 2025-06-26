# This is the input block for llama3. I use Tiny Shakespeare dataset for training and building the vocab.
# We will use character-level tokenization for simplicity.
# Each character will be embedded into a 128-dimensional vector.

import torch
from llama3 import device

class InputBlock:
    def __init__(self):
        # load the Tiny Shakespeare dataset
        with open('data/tiny_shakespeare.txt', 'r') as f:
            data = f.read()

        # create a set of unique characters in the dataset
        self.vocab = sorted(list(set(data)))
        self.vocab.extend(['<|begin_of_text|>', '<|end_of_text|>', '<|pad_id|>'])
        self.vocab_size = len(self.vocab)

        # create a mapping from characters to indices
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}

        # define the tensor markers for the start and end of text
        self.token_bos = torch.tensor([self.stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
        self.token_eos = torch.tensor([self.stoi['<|end_of_text|>']], dtype=torch.int, device=device)
        self.token_pad = torch.tensor([self.stoi['<|pad_id|>']], dtype=torch.int, device=device)


    # tokenizer encoder
    def encode(self, s):
        return [self.stoi[c] for c in s]

    # tokenizer decoder
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


    


