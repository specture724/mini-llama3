from llama3 import device
def test_input_block():
    from llama3.input_block import InputBlock

    # Create an instance of InputBlock
    input_block = InputBlock()

    assert input_block.encode("hello world") == [input_block.stoi[c] for c in "hello world"]

    assert input_block.decode(input_block.encode("hello world")) == "hello world"

def test_RMSNorm():
    import torch
    from llama3.RMSNorm import RMSNorm
    from llama3.param import params

    x = torch.randn((params.max_batch_size, params.max_seq_len, params.dim), device=device)  
    rms_norm = RMSNorm(dim=params.dim)  
    x_norm = rms_norm(x)  
    assert x_norm.shape == torch.Size([params.max_batch_size, params.max_seq_len, params.dim])