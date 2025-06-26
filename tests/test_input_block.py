from llama3.input_block import InputBlock


def test_input_block():

    # Create an instance of InputBlock
    input_block = InputBlock()

    assert input_block.encode("hello world") == [input_block.stoi[c] for c in "hello world"]

    assert input_block.decode(input_block.encode("hello world")) == "hello world"